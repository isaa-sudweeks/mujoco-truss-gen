from __future__ import annotations

import numpy as np

from mujoco_truss_gen.numerical import numerical_gradient


def test_numerical_gradient_estimates_scalar_function_gradient() -> None:
    values = np.array([0.25, -0.4], dtype=float)

    gradient = numerical_gradient(
        lambda inputs: float(inputs[0] ** 2 + 3.0 * inputs[1]),
        values,
        epsilon=1e-5,
    )

    np.testing.assert_allclose(gradient, [0.5, 3.0], rtol=1e-6, atol=1e-8)


def test_numerical_gradient_keeps_probes_inside_bounds() -> None:
    probes: list[np.ndarray] = []

    def objective(inputs: np.ndarray) -> float:
        probes.append(inputs.copy())
        return float(2.0 * inputs[0] - inputs[1])

    gradient = numerical_gradient(
        objective,
        np.array([1.0, -1.0]),
        epsilon=0.1,
        lower_bound=-1.0,
        upper_bound=1.0,
    )

    np.testing.assert_allclose(gradient, [2.0, -1.0])
    assert all(np.all(probe >= -1.0) and np.all(probe <= 1.0) for probe in probes)
