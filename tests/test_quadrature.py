# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Validate the quadrature representation of the polynomial kernel."""

import numpy as np
from utils import make_grid_poly_rational_chebyshev


def test_power_law_kernel_quadrature():
    """
    Compare quadrature-based and analytical autocorrelation functions.

    This function constructs the power-law ACID autocorrelation function
    using a rational Chebyshev quadrature over exponential kernels and
    compares it against the closed-form analytical expression.
    """
    # Power-law kernel parameters
    alpha = 3 / 2
    theta = 5.0
    a0 = 1.0

    # Quadrature order
    order = 80

    nstep = 1024
    times = np.arange(nstep, dtype=float)
    taus, weights = make_grid_poly_rational_chebyshev(order, theta, alpha)

    # Prune quadrature grid.
    mask = weights > weights.max() * 1e-34
    taus = taus[mask]
    weights = weights[mask]

    prefactor = a0 * (alpha - 1) / (2 * theta)
    quadrature_acf = (weights * prefactor) @ np.exp(-np.outer(1 / taus, times))

    analytical_acf = prefactor * (1 + abs(times) / theta) ** (-alpha)
    rel_err = np.abs(quadrature_acf - analytical_acf) / analytical_acf
    assert max(rel_err) <= 1e-12, "Quadrature does not reproduce analytical ACF within tolerance."
