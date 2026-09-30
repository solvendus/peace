import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Tuple, Union
from peace import units
from .symmetry import covariant_connection_commutator
from peace.state import resolve_state_dimensions

Array = Union[np.ndarray, jnp.ndarray]
DEFAULT_EIGENVECTOR_GRADIENT_REGULARIZATION_EV = 1.0e-3

@jax.custom_jvp
def _regularized_eigh(
    hamiltonian: Array,
    regularization: Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    del regularization
    energies, eigenvectors = jnp.linalg.eigh(hamiltonian)
    return energies, eigenvectors

@_regularized_eigh.defjvp
def _regularized_eigh_jvp(primals, tangents):
    hamiltonian, regularization = primals
    hamiltonian_dot, _ = tangents

    energies, eigenvectors = jnp.linalg.eigh(hamiltonian)
    hamiltonian_dot = 0.5 * (hamiltonian_dot + hamiltonian_dot.conj().T)
    rotated_dot = eigenvectors.conj().T @ hamiltonian_dot @ eigenvectors
    energies_dot = jnp.diag(rotated_dot).real
    gaps = energies[None, :] - energies[:, None]
    regularization = jnp.asarray(regularization, dtype=energies.dtype)
    inverse_gaps = gaps / (jnp.square(gaps) + jnp.square(regularization))
    inverse_gaps = inverse_gaps.at[
        jnp.diag_indices(energies.shape[0])
    ].set(0.0)
    eigenvectors_dot = eigenvectors @ (rotated_dot * inverse_gaps)

    return (
        (energies, eigenvectors),
        (energies_dot, eigenvectors_dot),
    )


def adiabatic_eigensystem(
    hamiltonian: Array,
    regularization: float = DEFAULT_EIGENVECTOR_GRADIENT_REGULARIZATION_EV,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _regularized_eigh(
        hamiltonian,
        jnp.asarray(regularization, dtype=jnp.asarray(hamiltonian).real.dtype),
    )


def free_displacement(Ra: Array, Rb: Array) -> Array:
    """Compute displacement vector between two positions without PBC."""
    return Ra - Rb

vmap_free_displacement = jax.vmap(free_displacement, in_axes=(0, 0))

def vmap_edges_fn(positions: Array,
                  senders: Array,
                  receivers: Array) -> Array:
    return free_displacement(positions[senders], positions[receivers])


def _symmetric_jacobian_from_upper(
    upper_jacobian: jnp.ndarray,
    n_states: int,
) -> jnp.ndarray:
    rows, cols = jnp.triu_indices(n_states)
    output = jnp.zeros(
        (n_states, n_states) + upper_jacobian.shape[1:],
        dtype=upper_jacobian.dtype,
    )
    output = output.at[rows, cols].set(upper_jacobian)
    return output.at[cols, rows].set(upper_jacobian)

def hamiltonian_with_coordinate_jacobian(model, params, graph):
    if getattr(model, 'include_soc', False):
        def complex_fn(positions):
            current = graph._replace(nodes=graph.nodes._replace(positions=positions))
            h, b = model.apply(params, current)
            return h, (h, b)
        dh, (h, b) = jax.jacfwd(complex_fn, has_aux=True)(graph.nodes.positions)
        return h, b, dh

    def upper_hamiltonian_fn(positions):
        current = graph._replace(
            nodes=graph.nodes._replace(positions=positions),
            edges=graph.edges._replace(shifts=vmap_edges_fn(
                positions, graph.senders, graph.receivers)))
        hamiltonian, connection = model.apply(params, current)
        rows, cols = jnp.triu_indices(hamiltonian.shape[0])
        return hamiltonian[rows, cols], (hamiltonian, connection)

    upper_jacobian, (hamiltonian, connection) = jax.jacrev(
        upper_hamiltonian_fn, has_aux=True)(graph.nodes.positions)
    jacobian = _symmetric_jacobian_from_upper(upper_jacobian, hamiltonian.shape[0])
    return hamiltonian, connection, jacobian

def hamiltonian_with_observed_derivatives(model, params, graph):
    if getattr(model, 'include_soc', False):
        h, b, dh = hamiltonian_with_coordinate_jacobian(model, params, graph)
        eps = DEFAULT_EIGENVECTOR_GRADIENT_REGULARIZATION_EV
        if model.use_au:
            eps /= units.Hartree
        e, u = adiabatic_eigensystem(h, eps)
        projected = jnp.einsum('ia,ijnc,jb->abnc', u.conj(), dh, u)
        rows, cols = jnp.triu_indices(h.shape[0])
        return h, b, e, u, projected[rows, cols]

    def hamiltonian_fn(positions):
        current = graph._replace(
            nodes=graph.nodes._replace(positions=positions),
            edges=graph.edges._replace(shifts=vmap_edges_fn(
                positions, graph.senders, graph.receivers)))
        h, connection = model.apply(params, current)
        return h, connection

    h, coordinate_pullback, connection = jax.vjp(
        hamiltonian_fn, graph.nodes.positions, has_aux=True)
    epsilon = DEFAULT_EIGENVECTOR_GRADIENT_REGULARIZATION_EV
    if getattr(model, 'use_au', False):
        epsilon /= units.Hartree
    energies, full_u = adiabatic_eigensystem(h, regularization=epsilon)
    n, _ = resolve_state_dimensions(
        getattr(model, 'n_states', h.shape[0]), h.shape[0])
    u = full_u[:, :n]
    rows, cols = np.triu_indices(n)
    left, right = u[:, rows].T, u[:, cols].T
    seeds = .5 * (jnp.einsum('pi,pj->pij', left, right)
                  + jnp.einsum('pi,pj->pij', right, left))
    projected_derivatives = jax.vmap(
        lambda seed: coordinate_pullback(seed)[0])(seeds)
    return h, connection, energies, full_u, projected_derivatives

def compute_fn_with_hamiltonian(model, params, graph):
    if getattr(model, 'include_soc', False):
        h, b, dh = hamiltonian_with_coordinate_jacobian(model, params, graph)
        eps = DEFAULT_EIGENVECTOR_GRADIENT_REGULARIZATION_EV
        if model.use_au:
            eps /= units.Hartree
        e, u = adiabatic_eigensystem(h, eps)
        commutator = covariant_connection_commutator(h, b)
        derivative = jnp.einsum('ia,ijnc,jb->abnc', u.conj(), dh + commutator, u)
        connection_part = jnp.einsum('ia,ijnc,jb->abnc', u.conj(), commutator, u)
        rows, cols = jnp.triu_indices(h.shape[0], 1)
        forces = -jnp.moveaxis(jnp.diagonal(derivative, axis1=0, axis2=1), -1, 0).real
        return ((e, forces, derivative[rows, cols],
                 jnp.moveaxis(connection_part[rows, cols], 0, 1)), h)

    h, connection, energies, full_u, projected_derivatives = (
        hamiltonian_with_observed_derivatives(model, params, graph))
    n, _ = resolve_state_dimensions(
        getattr(model, 'n_states', h.shape[0]), h.shape[0])
    u = full_u[:, :n]
    rows, cols = np.triu_indices(n)
    diagonal = np.flatnonzero(rows == cols)
    offdiagonal = np.flatnonzero(rows < cols)
    connection_diabatic = covariant_connection_commutator(h, connection)
    connection_projected = jnp.einsum(
        'km,kl...,ln->mn...', u, connection_diabatic, u)
    pair_rows, pair_cols = np.triu_indices(n, k=1)
    connection_sn = connection_projected[pair_rows, pair_cols]
    return ((energies[:n], -projected_derivatives[diagonal],
             projected_derivatives[offdiagonal] + connection_sn,
             jnp.moveaxis(connection_sn, 0, 1)), h)

def compute_fn(model, params, graph):
    return compute_fn_with_hamiltonian(model, params, graph)[0]
