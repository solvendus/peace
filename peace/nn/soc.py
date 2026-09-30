"""Opt-in spin-orbit PEACE, in SHARC's singlet / triplet-Ms ordering.
"""
import math
import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from peace import units
from .model import PEACEModel, default_config
from .quantity import adiabatic_eigensystem
from .symmetry import unpack_antisymmetric_connection, pack_antisymmetric_connection


def spin_root_indices(n_singlets, n_triplets):
    return np.r_[np.arange(n_singlets),
                 np.tile(np.arange(n_singlets, n_singlets + n_triplets), 3)]

def expand_spin_free(matrix, n_singlets, n_triplets):
    """Repeat the triplet block for Ms=-1,0,+1; preserve trailing axes."""
    n = n_singlets + 3 * n_triplets
    out = jnp.zeros((n, n) + matrix.shape[2:], dtype=matrix.dtype)
    out = out.at[:n_singlets, :n_singlets].set(matrix[:n_singlets, :n_singlets])
    for m in range(3):
        sl = slice(n_singlets + m * n_triplets, n_singlets + (m + 1) * n_triplets)
        out = out.at[sl, sl].set(matrix[n_singlets:, n_singlets:])
    return out

def soc_pairs(n_singlets, n_triplets):
    return (tuple((i, n_singlets + j) for i in range(n_singlets)
                  for j in range(n_triplets))
            + tuple((n_singlets + i, n_singlets + j)
                    for i in range(n_triplets) for j in range(i + 1, n_triplets)))

def cartesian_triplet_basis(dtype=jnp.complex128, *, spin_convention="sharc_v1"):
    if spin_convention not in ("sharc_v1", "legacy_cartesian_v1"):
        raise ValueError(f"Unknown SOC spin convention: {spin_convention}")
    q = 1.0 / math.sqrt(2.0)
    z = -1. if spin_convention == "sharc_v1" else 1.
    return jnp.asarray([[q, 0., -q], [-1j*q, 0., -1j*q], [0., z, 0.]], dtype=dtype)


def assemble_soc(vectors, n_singlets, n_triplets, *, spin_convention="sharc_v1"):
    dtype = jnp.result_type(vectors.dtype, 1j)
    n = n_singlets + 3 * n_triplets
    result = jnp.zeros((n, n), dtype=dtype)
    transform = cartesian_triplet_basis(dtype, spin_convention=spin_convention)
    q = 1.0 / math.sqrt(2.0)
    sx = jnp.asarray([[0., q, 0.], [q, 0., q], [0., q, 0.]], dtype=dtype)
    sy = jnp.asarray([[0., 1j*q, 0.], [-1j*q, 0., 1j*q], [0., -1j*q, 0.]], dtype=dtype)
    sz = jnp.diag(jnp.asarray([-1., 0., 1.], dtype=dtype))
    spin = jnp.stack([sx, sy, sz])
    if spin_convention == "sharc_v1":
        phase = jnp.asarray([1., -1., 1.], dtype=dtype)
        spin = phase[None, :, None] * spin * phase[None, None, :]
    for k, (i, j) in enumerate(soc_pairs(n_singlets, n_triplets)):
        jj = n_singlets + (j - n_singlets) + jnp.arange(3) * n_triplets
        if i < n_singlets:
            values = (1j * vectors[k]) @ transform
            result = result.at[i, jj].set(values)
            result = result.at[jj, i].set(values.conj())
        else:
            ii = n_singlets + (i - n_singlets) + jnp.arange(3) * n_triplets
            block = 1j * jnp.einsum('a,amn->mn', vectors[k], spin)
            result = result.at[ii[:, None], jj[None, :]].set(block)
            result = result.at[jj[:, None], ii[None, :]].set(block.conj().T)
    return result


class SpinOrbitHead(nn.Module):
    n_singlets: int
    n_triplets: int
    signature: tuple
    hidden_dim: int
    scale: float
    spin_convention: str

    @nn.compact
    def __call__(self, features, mask):
        pairs = soc_pairs(self.n_singlets, self.n_triplets)
        vectors = jnp.zeros((len(pairs), 3), dtype=features.dtype)
        hidden = nn.silu(nn.Dense(self.hidden_dim, name='scalar_hidden')(
            features.filter('0e').array))
        for irrep, same in [('1e', True), ('1o', False)]:
            indices = [k for k, (i, j) in enumerate(pairs)
                       if (self.signature[i] == self.signature[j]) == same]
            if not indices:
                continue
            basis = features.filter(irrep).array.reshape(features.shape[0], -1, 3)
            if basis.shape[1] == 0:
                raise ValueError(f'SOC readout requires {irrep} features')
            weights = nn.Dense(len(indices) * basis.shape[1], name=f'gates_{irrep}')(hidden)
            weights = weights.reshape(features.shape[0], len(indices), basis.shape[1])
            node_vectors = jnp.einsum('npc,ncd->npd', weights, basis)
            values = jnp.sum(node_vectors * jnp.asarray(mask).reshape(-1, 1, 1), axis=0)
            vectors = vectors.at[jnp.asarray(indices)].set(self.scale * values)
        return assemble_soc(vectors, self.n_singlets, self.n_triplets,
                            spin_convention=self.spin_convention)


class SOCPEACEModel(PEACEModel):
    """Two real PEACE sectors plus their rank-one SOC; opt-in only."""

    @property
    def electronic_dimension(self):
        return self.n_singlets + 3 * self.n_triplets

    @nn.compact
    def __call__(self, graph, *, return_components=False):
        if self.include_soc is not True:
            raise ValueError('SOCPEACEModel requires explicit include_soc: true')
        base = {key: getattr(self, key) for key in default_config()
                if key not in ('architecture', 'electronic_parity')}
        blocks, connections, features = [], [], []
        start = 0
        for label, count in [('singlet', self.n_singlets), ('triplet', self.n_triplets)]:
            kwargs = dict(base)
            kwargs.update(include_soc=False, n_singlets=None, n_triplets=None,
                          n_states=count, latent_n_states=count,
                          reflection_signature=tuple(self.reflection_signature[start:start+count]),
                          gap_average=(None if self.gap_average is None
                                       else self.gap_average[start:start+count]))
            h, b, feat = PEACEModel(**kwargs, name=label)(graph, return_features=True)
            blocks.append(h)
            connections.append(unpack_antisymmetric_connection(b, count))
            features.append(feat)
            start += count
        ns, nt = self.n_singlets, self.n_triplets
        h_sf = jnp.zeros((ns+nt, ns+nt), dtype=blocks[0].dtype)
        h_sf = h_sf.at[:ns, :ns].set(blocks[0]).at[ns:, ns:].set(blocks[1])
        b_sf = jnp.zeros((ns+nt, ns+nt) + connections[0].shape[2:], dtype=blocks[0].dtype)
        b_sf = b_sf.at[:ns, :ns].set(connections[0]).at[ns:, ns:].set(connections[1])
        h_soc = SpinOrbitHead(ns, nt, tuple(self.reflection_signature),
                             self.hamiltonian_hidden_dim, self.soc_scale, self.soc_spin_convention,
                             name='soc_head')(e3nn.concatenate(features), graph.nodes.mask)
        b_spin = pack_antisymmetric_connection(expand_spin_free(b_sf, ns, nt))
        if return_components:
            return h_sf, h_soc, b_spin
        return expand_spin_free(h_sf, ns, nt).astype(h_soc.dtype) + h_soc, b_spin

def spin_free_eigensystem(h_sf, n_singlets, *, use_au=False):
    eps = 1.e-3 / units.Hartree if use_au else 1.e-3
    es, us = adiabatic_eigensystem(h_sf[:n_singlets, :n_singlets], eps)
    et, ut = adiabatic_eigensystem(h_sf[n_singlets:, n_singlets:], eps)
    u = jnp.zeros_like(h_sf).at[:n_singlets, :n_singlets].set(us)
    u = u.at[n_singlets:, n_singlets:].set(ut)
    return jnp.concatenate([es, et]), u

def soc_quantities(model, params, graph, *, return_snac=False, return_basis=False):
    if getattr(model, 'include_soc', False) is not True:
        raise ValueError('soc_quantities requires include_soc: true')

    def spin_free_fn(positions):
        current = graph._replace(nodes=graph.nodes._replace(positions=positions))
        h_sf, h_soc, b_spin = model.apply(params, current, return_components=True)
        return h_sf, (h_soc, b_spin)

    h_sf, pullback, (h_soc, b_spin) = jax.vjp(
        spin_free_fn, graph.nodes.positions, has_aux=True)
    energies, u = spin_free_eigensystem(h_sf, model.n_singlets, use_au=model.use_au)
    seeds = jnp.einsum('ia,ja->aij', u, u)
    forces = -jax.vmap(lambda seed: pullback(seed)[0])(seeds)
    u_spin = expand_spin_free(u, model.n_singlets, model.n_triplets)
    soc_mch = u_spin.conj().T @ h_soc @ u_spin
    result = (energies, forces, soc_mch)
    if return_snac or return_basis:
        b_sf = unpack_antisymmetric_connection(
            b_spin, model.electronic_dimension)[:len(energies), :len(energies)]
        if return_snac:
            pairs = np.asarray(spin_free_pairs(model.n_singlets, model.n_triplets))
            left, right = u[:, pairs[:, 0]], u[:, pairs[:, 1]]
            off_seeds = jnp.einsum('ip,jp->pij', left, right)
            dh = jax.vmap(lambda seed: pullback(seed)[0])(off_seeds)
            b_mch = jnp.einsum('ip,ijad,jp->pad', left, b_sf, right)
            gaps = energies[pairs[:, 1]] - energies[pairs[:, 0]]
            snac = dh + gaps[:, None, None] * b_mch
            result += (snac,)
        if return_basis:
            result += (u, b_sf)
    return result

def spin_free_pairs(n_singlets, n_triplets):
    """Only same-multiplicity derivative couplings exist in the MCH basis."""
    return tuple((i, j) for lo, count in [(0, n_singlets), (n_singlets, n_triplets)]
                 for i in range(lo, lo+count) for j in range(i+1, lo+count))