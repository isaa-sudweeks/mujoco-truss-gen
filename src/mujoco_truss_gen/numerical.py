from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    *,
    epsilon: float,
    lower_bound: float | np.ndarray | None = None,
    upper_bound: float | np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a bounded scalar-function gradient with finite differences."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional input, got shape {values.shape}.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero")

    lower = np.broadcast_to(-np.inf if lower_bound is None else lower_bound, values.shape)
    upper = np.broadcast_to(np.inf if upper_bound is None else upper_bound, values.shape)
    if np.any(lower > upper):
        raise ValueError("lower_bound must not exceed upper_bound")

    gradient = np.zeros_like(values)
    for index in range(values.size):
        lower_value = max(values[index] - epsilon, lower[index])
        upper_value = min(values[index] + epsilon, upper[index])
        denominator = upper_value - lower_value
        if denominator <= 0.0:
            continue

        lower_values = values.copy()
        upper_values = values.copy()
        lower_values[index] = lower_value
        upper_values[index] = upper_value
        gradient[index] = (function(upper_values) - function(lower_values)) / denominator

    return gradient
