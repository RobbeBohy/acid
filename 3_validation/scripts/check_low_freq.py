#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Check the low-frequency part of the PSD."""

import argparse

import numpy as np
from numpy.typing import NDArray
from path import Path


def main():
    args = parse_args()
    run(args.zip_in)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether the low-frequency part of the PSD is well behaved."
    )
    parser.add_argument(
        "zip_in",
        type=Path,
        help="The zip file of the kernel to check.",
    )
    return parser.parse_args()


def run(path_kernel: Path):
    nstep = 1024

    unzipped_kernel = np.load(path_kernel)
    step_path = f"nstep{nstep:05d}/"
    psd = unzipped_kernel[step_path + "psd.npy"]
    freqs = unzipped_kernel[step_path + "freqs.npy"]

    check_low_freq(freqs, psd)


def check_low_freq(freqs: NDArray, psd: NDArray):
    """
    Check the low-frequency PSD against simple reference models.

    The low-frequency part of the spectrum is compared against two
    simple models:
        - Quadratic dependence on frequency
        - Power-law dependence on frequency with exponent 1/2

    The best fit of these two models is selected and required to meet
    fixed relative error thresholds over increasing frequency ranges:
        - The deviation should be less than 2.5 % for the first 10 frequency points.
        - The deviation should be less than 10.0 % for the first 20 frequency points.

    The relative noise in derived from 256 sequences is approximately 5 %.
    If this test passes,
    the low-frequency spectrum is sufficiently smooth for simple descriptions to
    remain valid over at least the first 20 frequency points.

    Parameters
    ----------
    freqs
         The array of frequencies for which to compute the spectrum.
    psd
        The power spectrum on the requested grid.
    """
    for nfit, threshold in (10, 0.025), (20, 0.100):
        my_freqs = freqs[:nfit]
        my_psd = psd[:nfit].copy()

        # Fit a simple quadratic, manually for robustness
        # my_psd -= my_psd.mean()
        my_psd -= my_psd[0]
        quad = my_freqs**2
        # quad -= quad.mean()
        par_quad = np.dot(quad, my_psd) / np.dot(quad, quad)
        fit_quad = par_quad * quad
        relerr_quad = float(np.linalg.norm(fit_quad - my_psd) / np.linalg.norm(my_psd))

        # Fit a polynomial function with exponent = 1/2, manually for robustness
        sqrt = np.sqrt(my_freqs)
        # sqrt -= sqrt.mean()
        par_sqrt = np.dot(sqrt, my_psd) / np.dot(sqrt, sqrt)
        fit_sqrt = par_sqrt * sqrt
        relerr_sqrt = float(np.linalg.norm(fit_sqrt - my_psd) / np.linalg.norm(my_psd))

        relerr_best_fit = min(relerr_quad, relerr_sqrt)

        if relerr_best_fit > threshold:
            raise ValueError(
                "The PSD is not approximated well by a simple model in the low-frequency domain:"
                f" {nfit=} {threshold=} {relerr_best_fit=}"
            )


if __name__ == "__main__":
    main()
