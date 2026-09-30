"""A single, state-consistent dot-product gauge for consecutive geometries."""
from itertools import product
import numpy as np

DEFAULT_NAC_DOT_THRESHOLD = 1.0e-7

def dot_product_state_phases(
    current_nacs,
    previous_aligned_nacs,
    pair_rows,
    pair_cols,
    n_states,
    eigenvector_dots,
    threshold=DEFAULT_NAC_DOT_THRESHOLD,
):
    current = np.asarray(current_nacs, dtype=np.float64)
    previous = np.asarray(previous_aligned_nacs, dtype=np.float64)
    rows = np.asarray(pair_rows, dtype=np.int64)
    cols = np.asarray(pair_cols, dtype=np.int64)
    eigen_dots = np.asarray(eigenvector_dots, dtype=np.float64)
    if current.shape != previous.shape or current.ndim < 2:
        raise ValueError("Current and previous NAC arrays must have matching (pair, ...) shape")
    if rows.shape != cols.shape or rows.ndim != 1 or current.shape[0] != rows.size:
        raise ValueError("NAC pair indices do not match the pair axis")
    if n_states < 1 or eigen_dots.shape != (n_states,):
        raise ValueError("Invalid electronic-state count or eigenvector dots")
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= n_states) or np.any(cols >= n_states) or np.any(rows == cols):
        raise ValueError("Invalid NAC state pair")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("NAC dot threshold must be finite and non-negative")
    if not all(np.isfinite(x).all() for x in (current, previous, eigen_dots)):
        raise FloatingPointError("Non-finite dot-product phase input")
    if n_states > 12:
        raise ValueError("Dot-product phase enumeration supports at most 12 states")

    dots = (
        np.einsum(
            "pk,pk->p", current.reshape(rows.size, -1), previous.reshape(rows.size, -1)
        )
        if rows.size else np.empty(0, dtype=np.float64)
    )
    evidence = np.where(np.abs(dots) > threshold, np.sign(dots), 0.0)
    weighted_dots = np.where(evidence != 0.0, dots, 0.0)
    patterns = np.asarray(list(product((1.0, -1.0), repeat=n_states)))
    pair_factors = patterns[:, rows] * patterns[:, cols]
    scores = pair_factors @ weighted_dots
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1.0e-12, atol=1.0e-14))
    signs = patterns[best[np.argmax(patterns[best] @ eigen_dots)]]
    return signs, signs[rows] * signs[cols], dots, evidence