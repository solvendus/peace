import jax.numpy as jnp
from peace.state import validate_n_states

def two_state_invariant_ci_distance(hamiltonian, eps=1.0e-12):
    if hamiltonian.shape[-2:] != (2, 2):
        raise ValueError(
            "two_state_invariant_ci_distance requires a 2x2 matrix")
    identity = jnp.eye(2, dtype=hamiltonian.dtype)
    mean_energy = jnp.trace(hamiltonian, axis1=-2, axis2=-1) / 2.0
    centered = hamiltonian - mean_energy[..., None, None] * identity
    trace_square = jnp.sum(centered * centered, axis=(-2, -1))
    energy_floor_sq = jnp.asarray(eps, dtype=hamiltonian.dtype)
    return (
        jnp.sqrt(2.0 * trace_square + energy_floor_sq)
        - jnp.sqrt(energy_floor_sq)
    )

def three_state_discriminant_from_hamiltonian(hamiltonian):

    if hamiltonian.shape[-2:] != (3, 3):
        raise ValueError(
            "three_state_discriminant_from_hamiltonian requires a 3x3 matrix")
    identity = jnp.eye(3, dtype=hamiltonian.dtype)
    mean_energy = jnp.trace(hamiltonian, axis1=-2, axis2=-1) / 3.0
    centered = hamiltonian - mean_energy[..., None, None] * identity
    trace_square = jnp.sum(centered * centered, axis=(-2, -1))
    c2 = -0.5 * trace_square
    c3 = jnp.linalg.det(centered)
    discriminant = -4.0 * c2 ** 3 - 27.0 * c3 ** 2
    return jnp.maximum(discriminant, 0.0)

def three_state_invariant_ci_distance(hamiltonian, eps=1.0e-12):
    if hamiltonian.shape[-2:] != (3, 3):
        raise ValueError(
            "three_state_invariant_ci_distance requires a 3x3 matrix")
    identity = jnp.eye(3, dtype=hamiltonian.dtype)
    mean_energy = jnp.trace(hamiltonian, axis1=-2, axis2=-1) / 3.0
    centered = hamiltonian - mean_energy[..., None, None] * identity
    trace_square = jnp.sum(centered * centered, axis=(-2, -1))
    discriminant = three_state_discriminant_from_hamiltonian(hamiltonian)

    energy_floor_sq = jnp.asarray(eps, dtype=hamiltonian.dtype)
    discriminant_floor = energy_floor_sq ** 3
    smooth_sqrt_discriminant = (
        jnp.sqrt(discriminant + discriminant_floor)
        - jnp.sqrt(discriminant_floor)
    )
    return smooth_sqrt_discriminant / (trace_square + energy_floor_sq)


def invariant_ci_distance(hamiltonian, eps=1.0e-12):
    n_states = validate_n_states(
        hamiltonian.shape[-1], context="Hamiltonian")
    if hamiltonian.shape[-2] != n_states:
        raise ValueError(
            "Hamiltonian must be square in its final two axes, got "
            f"{hamiltonian.shape[-2:]}")
    if n_states == 2:
        return two_state_invariant_ci_distance(hamiltonian, eps=eps)
    return three_state_invariant_ci_distance(hamiltonian, eps=eps)

def skew_symmetric_from_packed(packed, n_states, dtype=None):
    """Build an ``n_states x n_states`` skew matrix from packed upper entries."""
    packed = jnp.asarray(packed, dtype=dtype)
    rows, cols = jnp.triu_indices(n_states, k=1)
    skew = jnp.zeros((n_states, n_states), dtype=packed.dtype)
    skew = skew.at[rows, cols].set(packed)
    return skew - skew.T

def reflection_project_matrix(even_matrix, odd_matrix, reflection):
    even_matrix = jnp.asarray(even_matrix)
    odd_matrix = jnp.asarray(odd_matrix, dtype=even_matrix.dtype)
    reflection = jnp.asarray(reflection, dtype=even_matrix.dtype)
    reflected_even = jnp.einsum(
        "ia,ab...,jb->ij...", reflection, even_matrix, reflection)
    reflected_odd = jnp.einsum(
        "ia,ab...,jb->ij...", reflection, odd_matrix, reflection)
    return 0.5 * (
        even_matrix + reflected_even + odd_matrix - reflected_odd)

def project_vector_field_onto_rigid_motions(
    vectors,
    positions,
    node_mask,
    eps=1.0e-8,
):
    vectors = jnp.asarray(vectors)
    positions = jnp.asarray(positions, dtype=vectors.dtype)
    mask = jnp.asarray(node_mask, dtype=vectors.dtype).reshape(-1)
    masked_count = jnp.maximum(jnp.sum(mask), 1.0)
    centroid = jnp.sum(positions * mask[:, None], axis=0) / masked_count
    centered = (positions - centroid) * mask[:, None]

    axes = jnp.eye(3, dtype=vectors.dtype)
    n_nodes = positions.shape[0]
    translations = jnp.broadcast_to(
        axes[None, :, :], (n_nodes, 3, 3))
    rotations = jnp.stack(
        [jnp.cross(axis, centered) for axis in axes], axis=-1)
    modes = jnp.concatenate([translations, rotations], axis=-1)
    modes = modes * mask[:, None, None]
    mode_matrix = modes.reshape(-1, 6)

    vector_matrix = jnp.transpose(vectors, (0, 2, 1)).reshape(
        -1, vectors.shape[1])
    vector_matrix = vector_matrix * jnp.repeat(mask, 3)[:, None]
    gram = mode_matrix.T @ mode_matrix
    block_scales = jnp.stack([
        jnp.trace(gram[:3, :3]),
        jnp.trace(gram[3:, 3:]),
    ]) / 3.0
    ridge = eps * jnp.repeat(jnp.maximum(block_scales, 1.0), 3)

    augmented_modes = jnp.concatenate([
        mode_matrix, jnp.diag(jnp.sqrt(ridge)),
    ], axis=0)
    basis, _ = jnp.linalg.qr(augmented_modes, mode="reduced")
    cartesian_basis = basis[:mode_matrix.shape[0]]
    projected = cartesian_basis @ (cartesian_basis.T @ vector_matrix)
    projected = projected.reshape(n_nodes, 3, vectors.shape[1])
    projected = jnp.transpose(projected, (0, 2, 1))
    return projected * mask[:, None, None]

def project_vector_field_onto_internal_motions(
    vectors,
    positions,
    node_mask,
    eps=1.0e-8,
):
    vectors = jnp.asarray(vectors)
    mask = jnp.asarray(node_mask, dtype=vectors.dtype).reshape(-1)
    rigid = project_vector_field_onto_rigid_motions(
        vectors,
        positions,
        mask,
        eps=eps,
    )
    return (vectors - rigid) * mask[:, None, None]

def unpack_antisymmetric_connection(packed_connection, n_states):
    packed_connection = jnp.asarray(packed_connection)
    expected_pairs = n_states * (n_states - 1) // 2
    if packed_connection.shape[1] != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} connection pairs, got "
            f"{packed_connection.shape[1]}")

    rows, cols = jnp.triu_indices(n_states, k=1)
    packed_pair_first = jnp.moveaxis(packed_connection, 1, 0)
    connection = jnp.zeros(
        (n_states, n_states) + packed_pair_first.shape[1:],
        dtype=packed_connection.dtype,
    )
    connection = connection.at[rows, cols].set(packed_pair_first)
    connection = connection.at[cols, rows].set(-packed_pair_first.conj())
    return connection


def pack_antisymmetric_connection(connection):
    connection = jnp.asarray(connection)
    if connection.ndim < 2 or connection.shape[0] != connection.shape[1]:
        raise ValueError(
            "connection must begin with two equal electronic-state axes")
    rows, cols = jnp.triu_indices(connection.shape[0], k=1)
    pair_first = connection[rows, cols]
    return jnp.moveaxis(pair_first, 0, 1)

def covariant_connection_commutator(
    hamiltonian,
    packed_connection,
):
    hamiltonian = jnp.asarray(hamiltonian)
    packed_connection = jnp.asarray(
        packed_connection, dtype=hamiltonian.dtype)
    connection = unpack_antisymmetric_connection(
        packed_connection, hamiltonian.shape[0])
    commutator = (
        jnp.einsum("ik...,kj->ij...", connection, hamiltonian)
        - jnp.einsum("ik,kj...->ij...", hamiltonian, connection)
    )
    return 0.5 * (commutator + jnp.swapaxes(commutator.conj(), 0, 1))