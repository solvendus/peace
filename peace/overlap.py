import numpy as np

def validate_state_overlap(overlap, *, require_full_rank=False,
                           contraction_tolerance=1.e-6,
                           min_singular_value=1.e-6,
                           sharc_intruder_threshold=0.1):
    if (not np.isfinite(contraction_tolerance) or contraction_tolerance < 0
            or not np.isfinite(min_singular_value) or min_singular_value <= 0
            or not np.isfinite(sharc_intruder_threshold)
            or sharc_intruder_threshold < 0):
        raise ValueError('Overlap validation tolerances must be finite and valid')
    matrix = np.asarray(overlap, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError('Electronic-state overlap must be a nonempty square matrix')
    if not np.isfinite(matrix).all():
        raise FloatingPointError('Electronic-state overlap contains non-finite values')
    singular = np.linalg.svd(matrix, compute_uv=False)
    maximum, minimum = float(singular[0]), float(singular[-1])
    if maximum > 1. + contraction_tolerance:
        raise ValueError(
            'Electronic-state overlap is not a contraction: '
            f'sigma_max={maximum:.9g} exceeds 1 + {contraction_tolerance:g}. '
            'Check eigenvector frame normalization and phase signs.')
    if require_full_rank and minimum <= min_singular_value:
        raise ValueError(
            'Electronic-state overlap is nearly singular: '
            f'sigma_min={minimum:.9g}; cannot safely apply SHARC Loewdin '
            'orthogonalization. Reduce the nuclear step or enlarge the '
            'propagated electronic manifold; do not replace lost states by identity.')

    power = np.abs(matrix)**2
    intruder_scores = power.sum(axis=0) + power.sum(axis=1) - np.diag(power)
    intruders = np.flatnonzero(intruder_scores < sharc_intruder_threshold)
    if require_full_rank and intruders.size:
        raise ValueError(
            'Electronic-state overlap would trigger SHARC intruder-state '
            'identity repair: states (zero-based) '
            f'{intruders.tolist()}, scores {intruder_scores[intruders].tolist()} '
            f'< {sharc_intruder_threshold:g}. Reduce the nuclear step or '
            'reconsider the propagated electronic manifold.')
    defect = matrix.conj().T @ matrix - np.eye(matrix.shape[0])
    return {'singular_values':singular.tolist(),
            'min_singular_value':minimum,'max_singular_value':maximum,
            'orthogonality_defect_fro':float(np.linalg.norm(defect)),
            'maximum_subspace_deficit':float(max(0.,1.-minimum**2)),
            'sharc_intruder_scores':intruder_scores.tolist(),
            'full_rank_required':bool(require_full_rank)}
