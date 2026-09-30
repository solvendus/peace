from __future__ import annotations
from numbers import Integral
from typing import Any

SUPPORTED_STATE_COUNTS = (2, 3)
SUPPORTED_LATENT_STATE_COUNTS = tuple(
    range(min(SUPPORTED_STATE_COUNTS), 2 * max(SUPPORTED_STATE_COUNTS) + 1))

def validate_latent_n_states(n_states: Any, *, context: str = "PEACE") -> int:
    """Validate an electronic operator dimension, independently of labels."""
    if (isinstance(n_states, bool) or not isinstance(n_states, Integral)
            or int(n_states) not in SUPPORTED_LATENT_STATE_COUNTS):
        raise ValueError(
            f"{context} latent_n_states must be an integer from "
            f"{min(SUPPORTED_LATENT_STATE_COUNTS)} to {max(SUPPORTED_LATENT_STATE_COUNTS)}, "
            f"got {n_states!r}")
    return int(n_states)

def resolve_state_dimensions(n_states: Any, latent_n_states: Any = None) -> tuple[int, int]:
    observed = validate_n_states(n_states, context="Observed states")
    latent = (observed if latent_n_states is None else
              validate_latent_n_states(latent_n_states))
    if not observed <= latent <= 2 * observed:
        raise ValueError(
            f"latent_n_states must satisfy n_states <= latent_n_states <= 2*n_states "
            f"({observed} <= latent_n_states <= {2 * observed}), got {latent}")
    return observed, latent

def extend_energy_initialization(values, latent_n_states, *, n_states=None):
    values = list(values)
    latent = validate_latent_n_states(latent_n_states)
    if n_states is not None:
        observed, latent = resolve_state_dimensions(n_states, latent)
        if len(values) not in (observed, latent):
            raise ValueError(
                "Energy initialization must have one value per observed or latent state")
    if len(values) == latent:
        return values
    observed, latent = resolve_state_dimensions(len(values), latent)
    spacing = values[-1] - values[-2]
    return values + [
        values[-1] + step * spacing for step in range(1, latent - observed + 1)]

def validate_n_states(n_states: Any, *, context: str = "PEACE") -> int:
    """Return an exact supported state count without silent truncation."""
    if isinstance(n_states, bool):
        raise ValueError(f"{context} n_states must be 2 or 3, got {n_states!r}")
    if isinstance(n_states, Integral):
        value = int(n_states)
    else:
        try:
            value = int(n_states)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} n_states must be 2 or 3, got {n_states!r}"
            ) from exc
        if value != n_states:
            raise ValueError(
                f"{context} n_states must be an integer, got {n_states!r}"
            )
    if value not in SUPPORTED_STATE_COUNTS:
        raise ValueError(
            f"{context} supports exactly two or three electronic states, "
            f"got {value}"
        )
    return value

def state_pairs(n_states: Any) -> tuple[tuple[int, int], ...]:
    """Return the canonical packed upper-triangular state-pair order."""
    count = validate_n_states(n_states)
    return tuple(
        (state_i, state_j)
        for state_i in range(count)
        for state_j in range(state_i + 1, count)
    )

def n_state_pairs(n_states: Any) -> int:
    count = validate_n_states(n_states)
    return count * (count - 1) // 2

def state_pair_labels(n_states: Any) -> tuple[str, ...]:
    return tuple(f"S{i}-S{j}" for i, j in state_pairs(n_states))

def default_reflection_signature(n_states: Any) -> tuple[int, ...]:
    count = validate_n_states(n_states)
    if count == 2:
        return (1, -1)
    return (1, 1, -1)
