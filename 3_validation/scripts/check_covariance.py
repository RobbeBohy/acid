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
    mpl.rc_file(path_mplrc)
    lookup_table = np.load(path_codec)["midpoint"]
    unzipped_kernel = np.load(path_kernel)

    nstep = 1024
    nseq = 256

    with open(path_settings) as f0:
        settings = json.load(f0)

    step_path = f"nstep{nstep:05d}/"
    analytical_acf = unzipped_kernel[step_path + "acf.npy"]
    analytical_covar = sp.linalg.toeplitz(analytical_acf)

    nseed = settings["nseed"]
    with zipfile.ZipFile(path_kernel) as zf, zf.open("meta.json") as f:
        meta = json.load(f)

    std = np.sqrt(meta["var"])

    empirical_covar = np.zeros((nstep, nstep))
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
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    times = np.arange(analytical_eigenval.shape[0])

    axes[0].plot(times, empirical_eigenval, "r-", lw=2)
    axes[0].plot(times, analytical_eigenval, "k:", lw=2)

    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("Eigenvalue")

    rel_error = (empirical_eigenval - analytical_eigenval) / analytical_eigenval
    axes[1].axhline(0, color="k", lw=1, ls="--")
    axes[1].plot(times, rel_error, "r-", lw=2)
    axes[1].set_xlabel("Eigenvalue index")
    axes[1].set_ylabel("Relative error")

    fig.savefig(path_svg)


if __name__ == "__main__":
    main()
