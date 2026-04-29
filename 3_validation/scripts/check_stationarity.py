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
from path import Path


def main():
    args = parse_args()
    run(args.mplrc, args.zip_in, args.codec, args.settings, args.svg_var, args.svg_acf)


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
        "svg_var",
        type=Path,
        help="The output SVG path for the time-dependent variance plot.",
    )
    parser.add_argument(
        "svg_acf",
        type=Path,
        help="The output SVG path for the local ACF difference plot.",
    )
    return parser.parse_args()


def run(
    path_mplrc: Path,
    path_kernel: Path,
    path_codec: Path,
    path_settings: Path,
    path_svg_var: Path,
    path_svg_acf: Path,
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
        path_svg_var
            Output SVG path for the time-dependent variance plot.
        path_svg_acf
            Output SVG path for the autocorrelation difference plot.
    """
    mpl.rc_file(path_mplrc)
    nseq = 256
    nstep = 1024

    step_path = f"nstep{nstep:05d}/"

    lookup_table = np.load(path_codec)["midpoint"]
    unzipped_kernel = np.load(path_kernel)

    with open(path_settings) as f:
        settings = json.load(f)

    nseed = settings["nseed"]

    with zipfile.ZipFile(path_kernel) as zf, zf.open("meta.json") as f:
        meta = json.load(f)

    std = np.sqrt(meta["var"])

    plot_time_dependent_variance(
        unzipped_kernel,
        lookup_table,
        std,
        nstep,
        nseq,
        nseed,
        step_path,
        path_svg_var,
    )

    plot_acf_consistency(
        unzipped_kernel,
        lookup_table,
        std,
        nstep,
        nseq,
        nseed,
        step_path,
        path_svg_acf,
    )


def plot_time_dependent_variance(
    unzipped_kernel,
    lookup_table,
    std,
    nstep,
    nseq,
    nseed,
    step_path,
    path_svg,
):
    """Check that the variance is approximately constant in time."""
    var_t = np.zeros(nstep)

    for iseed in range(nseed):
        sample_path = step_path + f"nseq{nseq:04d}/sequences_{iseed:02d}.npy"
        cdfi = unzipped_kernel[sample_path]
        traj = lookup_table[cdfi] * std
        var_t += np.mean(traj**2, axis=0)

    var_t /= nseed
    var_t -= var_t[0]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(var_t)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Var(t) - Var(0)")

    fig.savefig(path_svg)


def plot_acf_consistency(
    unzipped_kernel,
    lookup_table,
    std,
    nstep,
    nseq,
    nseed,
    step_path,
    path_svg,
):
    """Check consistency of the autocorrelation across time windows."""
    # The maximum time lag that will be plotted
    max_lag = 150

    # The number of time lags that will be compared
    nbins = 4
    bins = np.array_split(np.arange(nstep), nbins)
    acf_blocks = np.zeros((nbins, max_lag))

    for ibin, b in enumerate(bins):
        acf = np.zeros(max_lag)

        for iseed in range(nseed):
            sample_path = f"{step_path}nseq{nseq:04d}/sequences_{iseed:02d}.npy"
            cdfi = unzipped_kernel[sample_path]
            traj = lookup_table[cdfi] * std

            block = traj[:, b]

            for lag in range(max_lag):
                acf[lag] += np.mean(block[:, : len(b) - lag] * block[:, lag:])

        acf_blocks[ibin] = acf / nseed

    fig, ax = plt.subplots(figsize=(6, 3))
    acf_diffs = acf_blocks - acf_blocks[0]

    for ibin in range(1, nbins):
        ax.plot(
            acf_diffs[ibin],
            lw=1.2,
            label=f"block {ibin} - block 0",
        )

    ax.axhline(0.0, color="k", ls="--", lw=0.8)

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    fig.savefig(path_svg)


if __name__ == "__main__":
    main()
