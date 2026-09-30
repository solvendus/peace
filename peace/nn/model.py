from typing import Any
import math
from numbers import Integral
import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp
from ml_collections import ConfigDict
from peace.nn.layers import (
    BesselEmbedding, DiabaticCouplingLayer, Linear,
    LocalChiralPseudoscalarSeed, LocalOutOfPlanePseudoscalar,
    NequIPConvolution, smooth_cutoff_envelope,
)
from peace.nn.attention import EquivariantSelfAttention
from peace.nn.symmetry import (
    project_vector_field_onto_internal_motions,
    project_vector_field_onto_rigid_motions, reflection_project_matrix,
)
from peace.nn.parity import ElectronicParity, resolve_parity
from peace.nn.structured_heads import StructuredConnectionHead
from peace.state import resolve_state_dimensions, extend_energy_initialization


class PEACEModel(nn.Module):
    graph_net_steps: int
    nonlinearities: Any
    n_elements: int
    hidden_irreps: str
    sh_irreps: str
    num_basis: int = 24
    r_max: float = 8.0
    cutoff_transition_width: float = 0.5
    radial_net_nonlinearity: str = "raw_swish"
    radial_net_n_hidden: int = 64
    radial_net_n_layers: int = 2
    n_neighbors: float = 5.0
    scalar_mlp_std: float = 4.0
    n_states: int = 3
    latent_n_states: Any = None
    include_soc: bool = False
    n_singlets: Any = None
    n_triplets: Any = None
    soc_scale: float = 0.01
    soc_spin_convention: str = "sharc_v1"
    gap_average: Any = None
    use_au: bool = False
    E0s: Any = None
    mapping: Any = None
    norm_stats: Any = None
    hamiltonian_hidden_dim: int = 128
    hamiltonian_mlp_layers: int = 2
    pair_encoder_n_layers: int = 3
    rigid_connection_scale: float = 0.1
    internal_connection_scale: float = 1.0
    lvc_oop_scale: float = 1.0
    hamiltonian_odd_scale: float = 1.0
    reflection_signature: tuple = ()
    attention_layers: int = 2
    attention_heads: int = 4
    attention_head_dim: int = 16
    attention_residual_scale: float = 0.1

    @property
    def uses_coordinate_nodes(self):
        return True

    @property
    def electronic_dimension(self):
        return resolve_state_dimensions(self.n_states, self.latent_n_states)[1]

    @nn.compact
    def __call__(self, graph, *, return_features=False):
        if self.include_soc:
            raise ValueError('Use model_from_config for include_soc models')
        n_states = self.electronic_dimension
        signature, _ = resolve_parity(
            n_states, signature=self.reflection_signature, n_observed_states=self.n_states)
        irreps = e3nn.Irreps(self.hidden_irreps)
        mask = jnp.asarray(graph.nodes.mask).reshape(-1, 1)
        edge_mask = jnp.asarray(graph.edges.mask).reshape(-1, 1)
        positions = graph.nodes.positions
        src, dst = graph.senders, graph.receivers
        dr = positions[src] - positions[dst]
        dr = jnp.where(edge_mask > 0, dr, jnp.array([1., 0., 0.], dtype=dr.dtype))
        distances = jnp.linalg.norm(dr, axis=-1)
        r_max = jnp.asarray(self.r_max, dtype=dr.dtype)
        inner = r_max - jnp.asarray(self.cutoff_transition_width, dtype=dr.dtype)
        radial = BesselEmbedding(
            count=self.num_basis, inner_cutoff=inner, outer_cutoff=r_max,
            name="radial_embedding")(distances)
        radial = radial * edge_mask
        edge_weight = edge_mask * smooth_cutoff_envelope(distances, inner, r_max)[:, None]
        harmonics = e3nn.spherical_harmonics(self.sh_irreps, dr, normalize=True)
        attrs = e3nn.IrrepsArray(f"{self.n_elements}x0e", graph.nodes.species)
        h = Linear(irreps_out=irreps, name="species_encoder")(attrs)
        n_even = sum(mul for mul, ir in irreps if ir == e3nn.Irrep("0e"))
        n_odd = sum(mul for mul, ir in irreps if ir == e3nn.Irrep("0o"))
        embedding = self.param(
            "species_embedding", nn.initializers.normal(.02), (self.n_elements, n_even))
        even_seed = graph.nodes.species @ embedding
        odd_seed = (LocalChiralPseudoscalarSeed(
            n_channels=n_odd, name="chiral_seed")(
                dr, dst, radial, edge_weight, positions.shape[0]).array
                    if n_odd else jnp.zeros((positions.shape[0], 0), dtype=dr.dtype))
        offset = even_offset = odd_offset = 0
        values = h.array
        for mul, ir in irreps:
            width = mul * ir.dim
            if ir == e3nn.Irrep("0e"):
                values = values.at[:, offset:offset + width].add(
                    even_seed[:, even_offset:even_offset + mul] * mask)
                even_offset += mul
            elif ir == e3nn.Irrep("0o"):
                values = values.at[:, offset:offset + width].add(
                    odd_seed[:, odd_offset:odd_offset + mul] * mask)
                odd_offset += mul
            offset += width
        h = e3nn.IrrepsArray(irreps, values) * mask
        for index in range(self.graph_net_steps):
            h = NequIPConvolution(
                hidden_irreps=irreps, use_sc=True,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_basis=self.num_basis, n_neighbors=self.n_neighbors,
                scalar_mlp_std=self.scalar_mlp_std,
                name=f"backbone_conv_{index}",
            )(h, attrs, harmonics, src, dst, radial) * mask

        h_connection = h
        for index in range(self.attention_layers):
            h = EquivariantSelfAttention(
                num_heads=self.attention_heads, head_dim=self.attention_head_dim,
                distance_scale=self.r_max, residual_scale=self.attention_residual_scale,
                name=f"hamiltonian_attention_{index}")(h, positions, mask)
            h_connection = EquivariantSelfAttention(
                num_heads=self.attention_heads, head_dim=self.attention_head_dim,
                distance_scale=self.r_max, residual_scale=self.attention_residual_scale,
                name=f"connection_attention_{index}")(h_connection, positions, mask)

        q_perp = LocalOutOfPlanePseudoscalar(
            scalar_mlp_std=self.scalar_mlp_std, name="out_of_plane_coordinate")(
                dr, dst, radial, edge_weight, positions.shape[0]) * mask
        even_features = jnp.concatenate(
            [h.filter("0e").array, h.filter("0o").array ** 2, q_perp ** 2], axis=-1)
        scalar_hidden = even_features
        for index in range(self.hamiltonian_mlp_layers):
            scalar_hidden = nn.silu(nn.Dense(
                self.hamiltonian_hidden_dim, name=f"diagonal_hidden_{index}")(scalar_hidden))
        diagonal = jnp.sum(nn.Dense(n_states, name="diagonal_output")(scalar_hidden) * mask, axis=0)
        if self.gap_average is not None:
            diagonal = diagonal + jnp.asarray(self.gap_average, dtype=diagonal.dtype)

        h_even, h_odd = DiabaticCouplingLayer(
            n_states=n_states, pair_hidden_dim=self.hamiltonian_hidden_dim,
            scalar_mlp_std=self.scalar_mlp_std,
            radial_net_n_hidden=self.radial_net_n_hidden,
            pair_encoder_n_layers=self.pair_encoder_n_layers,
            reflection_signature=signature,
            name="offdiagonal_readout",
        )(h, src, dst, radial, edge_weight, edge_sh=harmonics)
        rows, cols = jnp.triu_indices(n_states, 1)
        n_pairs = n_states * (n_states - 1) // 2
        oop_indices = ElectronicParity(signature).odd_pairs
        oop_pairs = jnp.zeros((n_pairs,), dtype=diagonal.dtype)
        if oop_indices:
            oop_hidden = nn.silu(nn.Dense(
                self.hamiltonian_hidden_dim, name="oop_coefficient_hidden")(even_features))
            oop_coefficients = nn.Dense(
                len(oop_indices) * q_perp.shape[-1], name="oop_coefficient_output")(oop_hidden)
            oop_coefficients = oop_coefficients.reshape(-1, len(oop_indices), q_perp.shape[-1])
            values = self.lvc_oop_scale * jnp.sum(
                q_perp[:, None, :] * oop_coefficients * mask[:, None, :], axis=(0, 2))
            oop_pairs = oop_pairs.at[jnp.asarray(oop_indices)].set(values)
        oop_matrix = jnp.zeros_like(h_odd).at[rows, cols].set(oop_pairs)
        h_odd = h_odd + oop_matrix + oop_matrix.T
        reflection = jnp.diag(jnp.asarray(signature, dtype=diagonal.dtype))
        h0 = reflection_project_matrix(
            jnp.diag(diagonal) + h_even,
            self.hamiltonian_odd_scale * h_odd, reflection)

        raw_rigid = StructuredConnectionHead(
            signature=signature, gated=False,
            name="rigid_connection_head")(h_connection, mask)
        raw_internal = StructuredConnectionHead(
            signature=signature, hidden_dim=self.hamiltonian_hidden_dim,
            zero_init_output=False, name="internal_connection_head")(h_connection, mask)
        rigid = self.rigid_connection_scale * project_vector_field_onto_rigid_motions(
            raw_rigid, positions, mask)
        internal = self.internal_connection_scale * project_vector_field_onto_internal_motions(
            raw_internal, positions, mask)
        self.sow("intermediates", "rigid_connection", rigid)
        self.sow("intermediates", "internal_connection", internal)
        self.sow("intermediates", "out_of_plane_pairs", oop_pairs)
        dummy = jnp.diag(jnp.arange(n_states, dtype=h0.dtype))
        h0 = jnp.where(jnp.sum(mask) > .5, h0, dummy)
        if return_features:
            return h0, rigid + internal, h
        return h0, rigid + internal


def default_config(architecture="peace"):
    if architecture != "peace":
        raise ValueError("PEACE supports only architecture='peace'; start a new run.")
    return ConfigDict({
        "architecture": "peace", "graph_net_steps": 4,
        "nonlinearities": {"e": "raw_swish", "o": "tanh"},
        "n_elements": 3,
        "hidden_irreps": "64x0e + 64x0o + 64x1o + 64x1e + 64x2e + 64x2o",
        "sh_irreps": "1x0e + 1x1o + 1x2e", "num_basis": 24,
        "r_max": 8.0, "cutoff_transition_width": 0.5,
        "radial_net_nonlinearity": "raw_swish", "radial_net_n_hidden": 64,
        "radial_net_n_layers": 2, "n_neighbors": 5.0, "scalar_mlp_std": 4.0,
        "n_states": 3, "latent_n_states": None, "gap_average": None, "use_au": False,
        "include_soc": False, "n_singlets": None, "n_triplets": None, "soc_scale": 0.01,
        "soc_spin_convention": "sharc_v1",
        "E0s": None, "mapping": None, "norm_stats": None,
        "hamiltonian_hidden_dim": 128, "hamiltonian_mlp_layers": 2,
        "pair_encoder_n_layers": 3,
        "rigid_connection_scale": 0.1, "internal_connection_scale": 1.0,
        "lvc_oop_scale": 1.0, "hamiltonian_odd_scale": 1.0,
        "electronic_parity": None, "reflection_signature": [],
        "attention_layers": 2, "attention_heads": 4, "attention_head_dim": 16,
        "attention_residual_scale": 0.1,
    })

def canonicalize_model_config(cfg):
    raw = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    if isinstance(raw.get("hamiltonian_odd_scale"), bool):
        raise ValueError("hamiltonian_odd_scale must be finite and between 0 and 1")
    canonical = default_config(raw.get("architecture", "peace"))
    unknown = sorted(set(raw) - set(canonical))
    if unknown:
        raise ValueError(f"Unknown PEACE configuration keys: {unknown}")
    if type(raw.get("include_soc", False)) is not bool:
        raise ValueError("include_soc must be an explicit YAML/Python boolean")
    canonical.update(raw)
    if canonical.include_soc:
        if canonical.soc_spin_convention not in ("sharc_v1", "legacy_cartesian_v1"):
            raise ValueError("Unknown soc_spin_convention; use sharc_v1 or legacy_cartesian_v1")
        for key in ("n_singlets", "n_triplets"):
            value = canonical[key]
            if type(value) is not int or value not in (2, 3):
                raise ValueError(f"{key} must be 2 or 3 for the current SOC sectors")
        observed = canonical.n_singlets + canonical.n_triplets
        if "n_states" in raw and raw["n_states"] != observed:
            raise ValueError("SOC n_states counts spin-free roots: n_singlets+n_triplets")
        if canonical.latent_n_states not in (None, observed):
            raise ValueError("Expanded latent sectors are not yet supported with SOC")
        latent = observed
        if not canonical.reflection_signature and canonical.electronic_parity is None:
            canonical.reflection_signature = (
                [1] * (canonical.n_singlets - 1) + [-1]
                + [1] * (canonical.n_triplets - 1) + [-1])
        if isinstance(canonical.soc_scale, bool) or not math.isfinite(canonical.soc_scale) or canonical.soc_scale <= 0:
            raise ValueError("soc_scale must be finite and positive in the model energy unit")
    else:
        if canonical.n_singlets is not None or canonical.n_triplets is not None:
            raise ValueError("Spin sectors require explicit include_soc: true")
        observed, latent = resolve_state_dimensions(canonical.n_states, canonical.latent_n_states)
    canonical.n_states = observed
    canonical.latent_n_states = latent
    signature, counts = resolve_parity(latent,
        canonical.electronic_parity, canonical.reflection_signature,
        n_observed_states=None if canonical.include_soc else observed)
    canonical.reflection_signature = list(signature)
    canonical.electronic_parity = list(counts)
    if canonical.gap_average is not None:
        if canonical.include_soc:
            if len(canonical.gap_average) != observed:
                raise ValueError("SOC gap_average requires one value per spin-free root")
        else:
            canonical.gap_average = extend_energy_initialization(
                canonical.gap_average, latent, n_states=observed)
        if not all(math.isfinite(value) for value in canonical.gap_average):
            raise ValueError("gap_average must contain finite initialization energies")
    for key in ("graph_net_steps", "n_elements", "num_basis", "radial_net_n_hidden",
                "radial_net_n_layers", "hamiltonian_hidden_dim", "hamiltonian_mlp_layers",
                "pair_encoder_n_layers", "attention_heads", "attention_head_dim"):
        value = canonical[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (isinstance(canonical.attention_layers, bool)
            or not isinstance(canonical.attention_layers, Integral)
            or canonical.attention_layers < 0):
        raise ValueError("attention_layers must be a nonnegative integer")
    for key in ("r_max", "cutoff_transition_width", "n_neighbors", "scalar_mlp_std"):
        if not math.isfinite(canonical[key]) or canonical[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if canonical.cutoff_transition_width > canonical.r_max:
        raise ValueError("cutoff_transition_width must not exceed r_max")
    for key in ("rigid_connection_scale", "internal_connection_scale", "lvc_oop_scale",
                "attention_residual_scale"):
        if not math.isfinite(canonical[key]) or canonical[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if (isinstance(canonical.hamiltonian_odd_scale, bool)
            or not math.isfinite(canonical.hamiltonian_odd_scale)
            or not 0.0 <= canonical.hamiltonian_odd_scale <= 1.0):
        raise ValueError("hamiltonian_odd_scale must be finite and between 0 and 1")
    layout = ElectronicParity(signature)
    required = ("0e",) + (("1o",) if layout.even_pairs else ()) + (("1e",) if layout.odd_pairs else ())
    if canonical.include_soc:
        required += ('1e', '1o')
    irreps = e3nn.Irreps(canonical.hidden_irreps)
    for irrep in required:
        if not any(mul > 0 and ir == e3nn.Irrep(irrep) for mul, ir in irreps):
            raise ValueError(f"hidden_irreps require {irrep} channels for this electronic parity")
    if any(ir.p != 1 for _, ir in e3nn.Irreps(canonical.sh_irreps) if ir.l % 2 == 0) or any(
            ir.p != -1 for _, ir in e3nn.Irreps(canonical.sh_irreps) if ir.l % 2):
        raise ValueError("Spherical harmonics must have natural spatial parity (-1)^l")
    return canonical

def model_from_config(cfg):
    values = canonicalize_model_config(cfg).to_dict()
    values.pop("architecture")
    values.pop("electronic_parity")
    values["reflection_signature"] = tuple(values["reflection_signature"])
    if values['include_soc']:
        from peace.nn.soc import SOCPEACEModel
        return SOCPEACEModel(**values)
    return PEACEModel(**values)