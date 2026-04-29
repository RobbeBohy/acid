#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Validate the quadrature representation of the polynomial kernel."""

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from utils import make_grid_poly_rational_chebyshev


def main():
    args = parse_args()
    run(args.mplrc, args.svg_sampling)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the quadrature approximation of the polynomial kernel."
    )
    parser.add_argument(
        "mplrc",
        type=Path,
        help="The matplotlibrc path.",
    )
    parser.add_argument(
        "svg_sampling",
        type=Path,
        help="The output SVG path.",
    )
    return parser.parse_args()


def run(path_mplrc: Path, path_svg: Path):
    """
    Compare quadrature-based and analytical autocorrelation functions.

    This function constructs the polynomial ACID autocorrelation function
    using a rational Chebyshev quadrature over exponential kernels and
    compares it against the closed-form analytical expression.

    Parameters
    ----------
    path_mplrc
        Path to the matplotlib configuration file.
    path_svg
        Output SVG path for the comparison plot.
    """
    mpl.rc_file(path_mplrc)

    # Polynomial kernel parameters
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

    quadrature_acf = 0

    for tau, weight in zip(taus, weights, strict=True):
        quadrature_acf += weight * a0 * (alpha - 1) / (2 * theta) * np.exp(-times / tau)

    analytical_acf = a0 * (alpha - 1) / (2 * theta) * (1 + abs(times) / theta) ** (-alpha)
    plot_sampling_bias(times, quadrature_acf, analytical_acf, path_svg)


def plot_sampling_bias(times, quadrature_acf, analytical_acf, path_svg):
    """Plot quadrature-based and analytical autocorrelation functions."""
    rel_err = np.abs(quadrature_acf - analytical_acf) / analytical_acf

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0.12, 0.12, 0.80, 0.80])

    ax.plot(times, quadrature_acf, "r-", lw=2, label="Quadrature")
    ax.plot(times, analytical_acf, "k:", lw=2, label="Analytical")
    ax.set_xlabel("Time lag")
    ax.set_ylabel("ACF")

    # Inset figure with relative errors
    ax_in = fig.add_axes([0.55, 0.45, 0.35, 0.35])
    ax_in.plot(times, rel_err, color="0.3", lw=1)
    ax_in.set_ylim(0, 1e-14)
    ax_in.set_xlabel("Time lag", fontsize=8)
    ax_in.set_ylabel("Relative error", fontsize=8)
    ax_in.tick_params(labelsize=8)

    fig.savefig(path_svg)


if __name__ == "__main__":
    main()
