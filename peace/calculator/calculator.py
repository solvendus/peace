from typing import Dict, List
import os
import warnings
import numpy as np
import jax
from peace.runtime import model_jit
import jax.numpy as jnp
from collections import namedtuple
import jraph
from peace import units
from peace.nn.quantity import (
    hamiltonian_with_observed_derivatives,
    _symmetric_jacobian_from_upper,
)
from peace.nn.io import atomic_energy_offset, load_model_artifacts
from peace.nn.symmetry import (
    covariant_connection_commutator,
    unpack_antisymmetric_connection,
)
from peace.graph import AtomicNumberTable, free_boundary_neighbor_list
from peace.state import n_state_pairs, validate_n_states
from .phase import DEFAULT_NAC_DOT_THRESHOLD, dot_product_state_phases

__all__ = [
    "PEACECalculator",
]

GraphNodes = namedtuple("Nodes", ["positions", "species", "mask"])
GraphEdges = namedtuple("Edges", ["shifts", "mask"])

if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])

class PEACECalculator:

    def __init__(
        self,
        modelpath: str,
        paramspath: str,
        atom_types: np.ndarray,
        safe_threshold = None,
        warn_on_state_mixing: bool = True,
        phase_dot_threshold: float = DEFAULT_NAC_DOT_THRESHOLD,
    ):

        model, params, model_configs = load_model_artifacts(modelpath, paramspath)
        self.model_configs = model_configs
        self.ztable = AtomicNumberTable.from_dict(model_configs['mapping'])
        self.n_total_states = validate_n_states(
            model_configs['n_states'], context="PEACE model"
        )
        self.n_latent_states = model_configs.get('latent_n_states', self.n_total_states)
        self.use_au = model_configs['use_au']
        self.cutoff = model_configs['r_max']
        self.has_shared_smooth_cutoff = True
        self.has_rigid_connection = True
        self.U_prev = None
        self.previous_aligned_nacs = None
        self.atom_types = atom_types
        self.energy_offset = atomic_energy_offset(model_configs, atom_types)
        self.nac_idx = np.triu_indices(self.n_total_states, k=1)
        self.n_pairs = n_state_pairs(self.n_total_states)
        self.n_atoms = len(self.atom_types)
        self.safe_threshold = (
            float(safe_threshold) if safe_threshold is not None
            else (1e-8 if jax.config.jax_enable_x64 else 1e-6)
        )
        self.warn_on_state_mixing = bool(warn_on_state_mixing)
        self.phase_dot_threshold = float(phase_dot_threshold)
        if (
            not np.isfinite(self.phase_dot_threshold)
            or self.phase_dot_threshold < 0.0
        ):
            raise ValueError(
                "phase_dot_threshold must be finite and non-negative"
            )
        self.last_overlap = None
        self.last_overlap_raw = None
        self.last_min_diag_overlap = None
        self.last_max_offdiag_overlap = None
        self.last_phase_signs = None
        self.last_pair_phase_factors = None
        self.last_nac_dot_products = None
        self.last_pair_phase_evidence = None

        def compute_fn(graph):
            h, connection, energies, full_u, upper_derivatives = (
                hamiltonian_with_observed_derivatives(model, params, graph))
            observed_u = full_u[:, :self.n_total_states]
            observed_dh = _symmetric_jacobian_from_upper(
                upper_derivatives, self.n_total_states)
            connection_dh = covariant_connection_commutator(h, connection)
            observed_connection_dh = jnp.einsum(
                'km,kl...,ln->mn...', observed_u, connection_dh, observed_u)
            connection_matrix = unpack_antisymmetric_connection(
                connection, h.shape[0])
            return (
                energies, full_u, observed_dh,
                observed_connection_dh, connection_matrix,
            )

        self.compute_fn = model_jit(compute_fn)

    @property
    def electronic_dimension(self):
        """Full H/B dimension; public energies and NACs use n_total_states."""
        return getattr(self, 'n_latent_states', self.n_total_states)

    def _reset_overlap_tracking(self) -> None:
        """Clear the eigenvector-dot overlap history."""
        self.U_prev = None
        self.last_overlap = None
        self.last_overlap_raw = None
        self.last_min_diag_overlap = None
        self.last_max_offdiag_overlap = None

    def _reset_nac_phase_tracking(self) -> None:
        """Clear adjacent-NAC dot-product phase history and diagnostics."""
        self.previous_aligned_nacs = None
        self.last_phase_signs = None
        self.last_pair_phase_factors = None
        self.last_nac_dot_products = None
        self.last_pair_phase_evidence = None

    def reset_phase_tracking(self) -> None:
        """Reset all temporal gauge state before starting a trajectory."""
        self._reset_overlap_tracking()
        self._reset_nac_phase_tracking()

    def _create_graphs(self, positions: np.ndarray) -> jraph.GraphsTuple:
        """Create padded Jraph tuple from coordinates."""
        senders, receivers = free_boundary_neighbor_list(positions, self.cutoff)
        graph = jraph.GraphsTuple(
            nodes=GraphNodes(
                positions=positions,
                species=jax.nn.one_hot(self.ztable.mapping(self.atom_types), len(self.ztable)),
                mask=np.ones_like(self.atom_types),
            ),
            edges=GraphEdges(
                shifts=np.zeros_like(positions[senders]),
                mask=np.ones(len(senders)),
            ),
            senders=senders,
            receivers=receivers,
            n_node=np.array([self.n_atoms]),
            n_edge=np.array([len(senders)]),
            globals=None,
        )

        padded_num_nodes = self.n_atoms + 1
        padded_num_edges = self.n_atoms * (self.n_atoms - 1) + 1
        return jraph.pad_with_graphs(graph, padded_num_nodes, padded_num_edges, 2)

    def _align_dot_phase(self, eigenvectors, raw_nacs_bohr_inv):
        """Align NACs and eigenvectors using only adjacent-frame dot products."""
        u = np.asarray(eigenvectors, dtype=np.float64)
        raw_nacs = np.asarray(raw_nacs_bohr_inv, dtype=np.float64)
        expected_nacs = (len(self.nac_idx[0]), self.n_atoms, 3)
        if u.shape != (self.electronic_dimension, self.electronic_dimension):
            raise ValueError("Electronic eigenvector frame has the wrong shape")
        if raw_nacs.shape != expected_nacs:
            raise ValueError(f"NAC shape {raw_nacs.shape}; expected {expected_nacs}")
        if not np.isfinite(u).all() or not np.isfinite(raw_nacs).all():
            raise FloatingPointError("Non-finite electronic phase input")

        if self.U_prev is None:
            signs = np.ones(u.shape[1], dtype=np.float64)
            pair_factors = np.ones(raw_nacs.shape[0], dtype=np.float64)
            dots = np.zeros(raw_nacs.shape[0], dtype=np.float64)
            evidence = np.zeros(raw_nacs.shape[0], dtype=np.float64)
            self.last_overlap_raw = None
            self.last_overlap = None
        else:
            overlap_raw = self.U_prev.T @ u
            signs, pair_factors, dots, evidence = dot_product_state_phases(
                raw_nacs,
                self.previous_aligned_nacs,
                self.nac_idx[0],
                self.nac_idx[1],
                u.shape[1],
                np.diag(overlap_raw),
                threshold=self.phase_dot_threshold,
            )
            self.last_overlap_raw = overlap_raw.copy()
            self.last_overlap = overlap_raw * signs[None, :]

        self.U_prev = u * signs[None, :]
        self.previous_aligned_nacs = raw_nacs * pair_factors[:, None, None]
        self.last_phase_signs = signs.copy()
        self.last_pair_phase_factors = pair_factors.copy()
        self.last_nac_dot_products = dots.copy()
        self.last_pair_phase_evidence = evidence.copy()

        if self.last_overlap is None:
            self.last_min_diag_overlap = None
            self.last_max_offdiag_overlap = None
            return pair_factors, np.eye(u.shape[1], dtype=np.float64)

        observed = self.last_overlap[:self.n_total_states, :self.n_total_states]
        if self.n_total_states > u.shape[1]:
            observed = self.last_overlap
        absolute = np.abs(observed)
        self.last_min_diag_overlap = float(np.min(np.diag(absolute)))
        self.last_max_offdiag_overlap = float(
            np.max(absolute - np.diag(np.diag(absolute)))
        )
        if self.warn_on_state_mixing and (
            self.last_max_offdiag_overlap > self.last_min_diag_overlap
            or self.last_min_diag_overlap < 0.8
        ):
            warnings.warn(
                "Strong adiabatic state mixing; eigenvector-dot overlap:\n"
                f"{observed}",
                RuntimeWarning,
                stacklevel=2,
            )
        return pair_factors, self.last_overlap.copy()

    def electronic_overlap(self) -> np.ndarray:
        if self.last_overlap is None:
            return np.eye(self.n_total_states, dtype=np.float64)
        return self.last_overlap[:self.n_total_states, :self.n_total_states].copy()

    def _read_observed_derivatives(self, derivative):
        """Read raw gradients and upper-triangular smoothed NACs."""
        grad = np.moveaxis(np.diagonal(derivative, axis1=0, axis2=1), -1, 0)
        row_idx, col_idx = self.nac_idx
        return grad, derivative[row_idx, col_idx]

    def _safe_gaps(self, gaps: np.ndarray) -> np.ndarray:
        """Avoid division by tiny gaps when recovering raw NACs."""
        signs = np.where(gaps >= 0.0, 1.0, -1.0)
        return np.where(np.abs(gaps) < self.safe_threshold, signs * self.safe_threshold, gaps)

    def calculate(self, sharc_coords: np.ndarray) -> Dict[str, List[np.ndarray]]:
        sharc_coords = np.asarray(sharc_coords)
        expected_coordinate_shape = (self.n_atoms, 3)
        if sharc_coords.shape != expected_coordinate_shape:
            raise ValueError(
                f"SHARC coordinates have shape {sharc_coords.shape}; "
                f"expected {expected_coordinate_shape}"
            )
        if not np.all(np.isfinite(sharc_coords)):
            raise FloatingPointError("SHARC coordinates contain non-finite values")
        positions = sharc_coords if self.use_au else sharc_coords * units.Bohr

        graph = self._create_graphs(positions)
        (
            energies,
            U_raw,
            observed_dh,
            observed_connection_dh,
            connection_matrix,
        ) = self.compute_fn(graph)

        relative_energies = np.asarray(energies, dtype=np.float64)
        energies = relative_energies + self.energy_offset
        U_raw = np.array(U_raw)
        observed_dh = np.array(observed_dh)
        observed_connection_dh = np.array(observed_connection_dh)
        connection_matrix = np.array(connection_matrix)

        expected_model_shapes = {
            "energies": (self.electronic_dimension,),
            "eigenvectors": (self.electronic_dimension, self.electronic_dimension),
        }
        model_values = {
            "energies": energies,
            "eigenvectors": U_raw,
        }
        for name, expected_shape in expected_model_shapes.items():
            value = model_values[name]
            if value.shape != expected_shape:
                raise ValueError(
                    f"PEACE {name} has shape {value.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"PEACE returned non-finite values in {name}"
                )
        derivative_values = {
            "observed Hamiltonian derivative": (
                observed_dh, self.n_total_states),
            "observed connection derivative": (
                observed_connection_dh, self.n_total_states),
            "diabatic connection": (
                connection_matrix, self.electronic_dimension),
        }
        for name, (value, state_dimension) in derivative_values.items():
            valid_shape = (
                value.ndim == 4
                and value.shape[:2] == (state_dimension, state_dimension)
                and value.shape[2] >= self.n_atoms
                and value.shape[3] == 3
            )
            if not valid_shape:
                raise ValueError(
                    f"PEACE {name} has shape {value.shape}; expected "
                    f"({state_dimension}, {state_dimension}, "
                    f">={self.n_atoms}, 3)"
                )
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"PEACE returned non-finite values in {name}"
                )
        if not (
            observed_dh.shape == observed_connection_dh.shape
            and observed_dh.shape[2:] == connection_matrix.shape[2:]
        ):
            raise ValueError(
                "PEACE derivative tensors have inconsistent atom axes: "
                f"observed={observed_dh.shape}, connection_derivative="
                f"{observed_connection_dh.shape}, connection={connection_matrix.shape}"
            )
        connection_matrix = connection_matrix[:, :, :self.n_atoms, :]

        energies = energies[:self.n_total_states]
        relative_energies = relative_energies[:self.n_total_states]
        grad, geometric_smoothed_nacs = self._read_observed_derivatives(
            observed_dh)
        _, connection_smoothed_nacs = self._read_observed_derivatives(
            observed_connection_dh)
        smoothed_nacs = (
            geometric_smoothed_nacs + connection_smoothed_nacs
        )
        grad = np.array(grad)[:, :self.n_atoms, :]
        smoothed_nacs = np.array(smoothed_nacs)[:, :self.n_atoms, :]
        geometric_smoothed_nacs = np.array(
            geometric_smoothed_nacs)[:, :self.n_atoms, :]
        connection_smoothed_nacs = np.array(
            connection_smoothed_nacs)[:, :self.n_atoms, :]

        rows, cols = np.triu_indices(self.n_total_states, k=1)
        gaps = relative_energies[rows] - relative_energies[cols]
        gaps = gaps[:, None, None]
        gaps_safe = self._safe_gaps(gaps)
        nacs = smoothed_nacs / gaps_safe
        geometric_nacs = geometric_smoothed_nacs / gaps_safe
        connection_nacs = connection_smoothed_nacs / gaps_safe

        energies_ha = energies if self.use_au else energies / units.Hartree
        grad_ha_bohr = grad if self.use_au else grad * (units.Bohr / units.Hartree)
        nacs_bohr_inv = nacs if self.use_au else nacs * units.Bohr
        geometric_nacs_bohr_inv = (
            geometric_nacs if self.use_au else geometric_nacs * units.Bohr
        )
        connection_nacs_bohr_inv = (
            connection_nacs if self.use_au else connection_nacs * units.Bohr
        )
        connection_matrix_bohr_inv = (
            connection_matrix
            if self.use_au
            else connection_matrix * units.Bohr
        )

        pair_factors, _ = self._align_dot_phase(U_raw, nacs_bohr_inv)
        pair_factors = pair_factors[:, None, None]
        nacs_bohr_inv = nacs_bohr_inv * pair_factors
        geometric_nacs_bohr_inv *= pair_factors
        connection_nacs_bohr_inv *= pair_factors
        smoothed_nacs *= pair_factors
        electronic_overlap = self.electronic_overlap()

        peace_output = {
            "energy": np.asarray(energies_ha),
            "gradients": np.asarray(grad_ha_bohr),
            "nacs": np.asarray(nacs_bohr_inv),
            "snacs": np.asarray(smoothed_nacs if self.use_au else smoothed_nacs * (units.Bohr / units.Hartree)),
            "geometric_nacs": np.asarray(geometric_nacs_bohr_inv),
            "connection_nacs": np.asarray(connection_nacs_bohr_inv),
            "diabatic_connection": np.asarray(connection_matrix_bohr_inv),
            "overlap": np.asarray(electronic_overlap),
        }

        expected_pairs = n_state_pairs(self.n_total_states)
        expected_output_shapes = {
            "energy": (self.n_total_states,),
            "gradients": (self.n_total_states, self.n_atoms, 3),
            "nacs": (expected_pairs, self.n_atoms, 3),
            "snacs": (expected_pairs, self.n_atoms, 3),
            "geometric_nacs": (expected_pairs, self.n_atoms, 3),
            "connection_nacs": (expected_pairs, self.n_atoms, 3),
            "diabatic_connection": (
                self.electronic_dimension,
                self.electronic_dimension,
                self.n_atoms,
                3,
            ),
            "overlap": (self.n_total_states, self.n_total_states),
        }
        for name, expected_shape in expected_output_shapes.items():
            value = peace_output[name]
            if value.shape != expected_shape:
                raise ValueError(
                    f"PEACE {name} has shape {value.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"PEACE returned non-finite values in {name}"
                )

        return peace_output

    def get_qm(self, peace_output: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Format PEACE outputs into the specific nested list dictionary
        expected by the SHARC Python API.
        """
        states = self.n_total_states
        expected_pairs = n_state_pairs(states)
        expected_shapes = {
            "energy": (states,),
            "gradients": (states, self.n_atoms, 3),
            "nacs": (expected_pairs, self.n_atoms, 3),
            "snacs": (expected_pairs, self.n_atoms, 3),
            "overlap": (states, states),
        }
        for name, expected_shape in expected_shapes.items():
            if name not in peace_output:
                raise KeyError(f"PEACE output is missing {name!r}")
            value = np.asarray(peace_output[name])
            if value.shape != expected_shape:
                raise ValueError(
                    f"PEACE {name} has shape {value.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"PEACE returned non-finite values in {name}"
                )
        qm_out = {}
        qm_out["h"] = np.diag(np.array(peace_output["energy"], dtype=complex))
        qm_out["grad"] = np.ascontiguousarray(
            peace_output["gradients"], dtype=np.float64)
        nacs_v = peace_output["nacs"]
        nacs_m = np.zeros((states, states, self.n_atoms, 3))
        nacs_m[self.nac_idx] = nacs_v
        nacs_m -= np.transpose(nacs_m, axes=(1, 0, 2, 3))
        qm_out["nacdr"] = nacs_m
        qm_out["overlap"] = np.asarray(
            peace_output["overlap"], dtype=np.complex128
        )

        return qm_out