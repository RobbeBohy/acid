#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Check empirical second-order stationarity of sampled trajectories."""

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
    run(args.mplrc, args.zip_in, args.codec, args.settings, args.svg_qq)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check time-translation invariance of empirical second moments."
    )
    parser.add_argument(
        "mplrc",
        type=Path,
        help="The matplotlibrc path.",
    )
    parser.add_argument(
        "zip_in",
        type=Path,
        help="The zip file containing sampled sequences.",
    )
    parser.add_argument(
        "codec",
        type=Path,
        help="The codec zip used to decode integer sequences.",
    )
    parser.add_argument(
        "settings",
        type=Path,
        help="The settings.json file.",
    )
    parser.add_argument(
        "svg_qq",
        type=Path,
        help="The output SVG path for the QQ plot.",
    )
    return parser.parse_args()


def run(
    path_mplrc: Path,
    path_kernel: Path,
    path_codec: Path,
    path_settings: Path,
    path_svg_qq: Path,
):
    """
    Check empirical second-order stationarity of sampled trajectories.

    Parameters
    ----------
    path_mplrc
        Path to the matplotlib configuration file.
    path_kernel
        ZIP archive of the desired kernel.
    path_codec
        Codec ZIP used to decode integer sequences to floating-point values.
    path_settings
        JSON file specifying nsteps, nseqs, and nseeds.
    path_svg_qq
        Output SVG path for the QQ plot.
    """
    mpl.rc_file(path_mplrc)
    lookup_table = np.load(path_codec)["midpoint"]
    unzipped_kernel = np.load(path_kernel)

    with open(path_settings) as f:
        settings = json.load(f)

    nseed = settings["nseed"]
    # The shortest trajectory length strongly amplifies finite-sample effects and is
    # therefore not representative for assessing convergence and bias in this check.
    nstep = settings["nsteps"][1]
    nseq = settings["nseqs"][-1]
    step_path = f"nstep{nstep:05d}/"

    with zipfile.ZipFile(path_kernel) as zf, zf.open("meta.json") as f:
        meta = json.load(f)

    std = np.sqrt(meta["var"])

    slice_length = 200
    init_slice = slice(0, slice_length)
    final_slice = slice(nstep - slice_length, nstep)

    init_traj = []
    final_traj = []

    for iseed in range(nseed):
        sample_path = f"{step_path}nseq{nseq:04d}/sequences_{iseed:02d}.npy"
        cdfi = unzipped_kernel[sample_path]
        traj = lookup_table[cdfi] * std

        init_traj.append(traj[:, init_slice].ravel())
        final_traj.append(traj[:, final_slice].ravel())

    init_samples = np.concatenate(init_traj)
    final_samples = np.concatenate(final_traj)

    init_samples.sort()
    final_samples.sort()

    plot_qq(init_samples, final_samples, path_svg_qq)


def plot_qq(init_samples, final_samples, path_svg):
    """Generates a QQ plot between the early-time samples and the late-time sample."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(init_samples, final_samples, rasterized=True, s=1, alpha=0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Early-time quantiles")
    ax.set_ylabel("Late-time quantiles")

    wasserstein_dist = sp.stats.wasserstein_distance(init_samples, final_samples)
    ax.set_title(f"Wasserstein distance = {wasserstein_dist:.5e}")

    fig.savefig(path_svg)


if __name__ == "__main__":
    main()
