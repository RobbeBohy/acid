#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Validation of the encoding and decoding scheme."""

import argparse
from runpy import run_path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from kernels import compute
from numpy.typing import NDArray
from path import Path
from stepup.core.api import amend
from utils import compute_amplitudes, lookup_integer

SEQ_DTYPE = np.uint16
IMAX = np.iinfo(SEQ_DTYPE).max + 1


def main():
    args = parse_args()
    run(args.mplrc, args.kernel_name, args.codec, args.svg_codec, args.svg_diff)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the encoding and decoding scheme.")
    parser.add_argument(
        "mplrc",
        type=Path,
        help="The matplotlibrc path.",
    )
    parser.add_argument(
        "codec",
        type=Path,
        help="The codec zip containing the encoding and decoding lookup tables.",
    )
    parser.add_argument(
        "kernel_name",
        type=str,
        help="The kernel name.",
    )
    parser.add_argument(
        "svg_codec",
        type=Path,
        help="The output SVG path for the ACF convergence plot.",
    )
    parser.add_argument(
        "svg_diff",
        type=Path,
        help="The output SVG path for the ACF difference plot.",
    )
    return parser.parse_args()


def run(
    path_mplrc: Path, kernel_name: str, path_codec: Path, path_svg_codec: Path, path_svg_diff: Path
):
    """
    Validate numerical fidelity of the encoding and decoding scheme.

    Parameters
    ----------
    path_mplrc
        Path to the matplotlib configuration file.
    kernel_name
        Name of the kernel.
    path_codec
        Codec ZIP containing the encoding and decoding lookup tables.
    path_svg_codec
        Output SVG path for the ACF convergence plot.
    path_svg_diff
        Output SVG path for the ACF difference plot.
    """
    mpl.rc_file(path_mplrc)

    path_py = f"../1_dataset/kernels/{kernel_name}.py"
    amend(inp=path_py)
    terms = run_path(path_py)["terms"]

    lookup_table_midpoint = np.load(path_codec)["midpoint"]
    lookup_table_boundary = np.load(path_codec)["boundary"]
    lengths = [
        (2**10),
        (2**12),
        (2**14),
        (2**16),
        (2**18),
        (2**20),
        (2**22),
        (2**24),
        (2**25),
    ]

    nstep = 2**25
    nseq = 1

    # We only need the time = 0 to determine the variance
    times = np.arange(16, dtype=float)
    freqs = np.fft.rfftfreq(16)
    _, acf_analyt, _, _, _, _, _ = compute(terms, freqs, times)
    std = np.sqrt(acf_analyt[0])

    float_acfs = np.zeros((len(lengths), lengths[-1]))
    codec_acfs = np.zeros((len(lengths), lengths[-1]))

    for term in terms:
        seed = np.frombuffer(f"{kernel_name}_{term}".encode("ascii"), dtype=np.uint8)
        rng = np.random.default_rng(seed)
        float_traj = term.sample(nseq, nstep, rng)
        encoded_traj = lookup_integer(float_traj, std, lookup_table_boundary)

        if encoded_traj.max() >= IMAX:
            raise ValueError(f"ppfi exceeds {IMAX - 1}")
        if encoded_traj.min() < 0:
            raise ValueError("Negative ppfi values found")
        encoded_traj = encoded_traj.astype(SEQ_DTYPE)
        decoded_traj = lookup_table_midpoint[encoded_traj] * std

        for ilength, length in enumerate(lengths):
            float_acfs[ilength, :length] += compute_acf(float_traj[:, :length])
            codec_acfs[ilength, :length] += compute_acf(decoded_traj[:, :length])

    plot_convergence(lengths, float_acfs, codec_acfs, path_svg_codec)
    plot_diffs(float_acfs, codec_acfs, path_svg_diff)


def compute_acf(traj: NDArray[float]) -> NDArray[float]:
    """
    Compute the autocorrelation function (ACF) from a batch of sequences.

    Parameters
    ----------
    traj
        The input trajectories, which is an array with shape ``(nindep, nstep)``.
        Each row is a time-dependent sequence.

    Returns
    -------
    acf
        The autocorrelation function.
    """
    psd = compute_amplitudes(traj)
    return np.fft.irfft(psd)


def plot_convergence(lengths, float_acfs, codec_acfs, path_svg_codec):
    """Plot convergence of autocorrelation estimates with trajectory length."""

    # A few arbitrary lags are selected to be checked
    lags_to_check = [1, 2, 5, 10, 20, 50, 100]

    fig, ax = plt.subplots(figsize=(6, 4))

    for ell in lags_to_check:
        vals_float = [acf_f[ell] for acf_f in float_acfs]
        vals_codec = [acf_c[ell] for acf_c in codec_acfs]

        ax.plot(lengths, vals_codec, "r")
        ax.plot(lengths, vals_float, "k:")

        ax.hlines(
            vals_float[-1],
            xmin=0,
            xmax=lengths[-1],
            colors="k",
            linestyles="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("ACF")

    fig.savefig(path_svg_codec)


def plot_diffs(float_acfs, codec_acfs, path_svg_diff):
    """Plot differences between encoded and floating-point autocorrelations."""
    fig, ax = plt.subplots(figsize=(6, 3))
    diff = codec_acfs[-1] - float_acfs[-1]
    lag = np.arange(len(float_acfs[-1]))

    ax.plot(lag, diff, color="0.3", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Time lag")
    ax.set_ylabel("ACF diff")

    fig.savefig(path_svg_diff)


if __name__ == "__main__":
    main()
