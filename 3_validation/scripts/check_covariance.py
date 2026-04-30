#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Validate the covariance matrix and check against bias"""

import argparse
import json
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from path import Path


def main():
    args = parse_args()
    run(args.mplrc, args.zip_in, args.codec, args.settings, args.svg_covar)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the empirical covariance matrix to detect systematic bias."
    )
    parser.add_argument(
        "mplrc",
        type=Path,
        help="The matplotlibrc path.",
    )
    parser.add_argument(
        "zip_in",
        type=Path,
        help="The zip file of the kernel to check.",
    )
    parser.add_argument(
        "codec",
        type=Path,
        help="The codec zip to decode the integers to floats",
    )
    parser.add_argument(
        "settings",
        type=Path,
        help="The settings.json file.",
    )
    parser.add_argument(
        "svg_covar",
        type=Path,
        help="The output SVG path for the covariance matrices.",
    )
    return parser.parse_args()


def run(path_mplrc: Path, path_kernel: Path, path_codec: Path, path_settings: Path, path_svg: Path):
    """
    Validate the empirical covariance against the covariance of the analytical kernel.

    Parameters
    ----------
    path_mplrc
        Path to the matplotlib configuration file.
    path_kernel
        ZIP archive of the kernel.
    path_codec
        Codec ZIP used to decode integer sequences to floating-point values.
    path_settings
        JSON file specifying the number of steps, number of sequences,
        and number of seeds used for sampling.
    path_svg
        Output SVG path for the covariance eigenvalue comparison plot.
    """
    mpl.rc_file(path_mplrc)
    lookup_table = np.load(path_codec)["midpoint"]
    unzipped_kernel = np.load(path_kernel)

    with open(path_settings) as f:
        settings = json.load(f)

    nseed = settings["nseed"]
    nseq = settings["nseqs"][-1]
    # The shortest trajectory length strongly amplifies finite sample effects and is
    # therefore not representative for assessing convergence and bias in this check.
    nstep = settings["nsteps"][1]

    step_path = f"nstep{nstep:05d}/"
    analytical_acf = unzipped_kernel[step_path + "acf.npy"]
    analytical_covar = sp.linalg.toeplitz(analytical_acf)

    with zipfile.ZipFile(path_kernel) as zf, zf.open("meta.json") as f:
        meta = json.load(f)

    std = np.sqrt(meta["var"])

    empirical_acf = np.zeros(nstep)

    for iseed in range(nseed):
        sample_path = f"nstep{nstep:05d}/nseq{nseq:04d}/sequences_{iseed:02d}.npy"
        cdfi = unzipped_kernel[sample_path]
        traj = lookup_table[cdfi] * std

        for step in range(nstep):
            empirical_acf[step] += np.mean(traj[:, : nstep - step] * traj[:, step:])

    empirical_acf /= nseed
    empirical_covar = sp.linalg.toeplitz(empirical_acf)

    analytical_eigenval = np.linalg.eigvalsh(analytical_covar)[::-1]
    empirical_eigenval = np.linalg.eigvalsh(empirical_covar)[::-1]

    plot_covariance_eigenvals(analytical_eigenval, empirical_eigenval, path_svg)


def plot_covariance_eigenvals(analytical_eigenval, empirical_eigenval, path_svg):
    """Plot analytical and empirical covariance eigenvalue spectra."""
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.arange(analytical_eigenval.shape[0])

    ax.plot(times, empirical_eigenval, "r-", lw=2)
    ax.plot(times, analytical_eigenval, "k:", lw=2)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")

    rel_error = (empirical_eigenval - analytical_eigenval) / analytical_eigenval

    # Inset plot with relative errors
    ax_in = fig.add_axes([0.55, 0.45, 0.35, 0.35])
    ax_in.axhline(0, color="k", lw=1, ls="--")
    ax_in.plot(times, rel_error, color="0.3", lw=1)
    ax_in.set_xlabel("Eigenvalue index")
    ax_in.set_ylabel("Relative error")

    fig.savefig(path_svg)


if __name__ == "__main__":
    main()
