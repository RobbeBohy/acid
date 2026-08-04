#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later
"""Tabulate the stationarity CvM test results into a CSV."""

import argparse

import numpy as np
from path import Path


def main():
    args = parse_args()
    run(args.stat_npz_paths, args.csv_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Tabulate stationarity test results into a CSV.")
    parser.add_argument(
        "stat_npz_paths",
        type=Path,
        nargs="+",
        help="The paths to the stationarity NPZ files.",
    )
    parser.add_argument("csv_out", type=Path, help="Output CSV path for the stationarity results.")
    return parser.parse_args()


def run(paths_stat_npz, path_csv):
    with open(path_csv, "w") as fh:
        for stat_npz in paths_stat_npz:
            kernel = stat_npz.stem.split("_")[0]
            data = np.load(stat_npz)
            statistics = data["statistics"]
            pvalues = data["pvalues"]
            fields = [f'"{kernel}"']
            for statistic, pvalue in zip(statistics, pvalues, strict=True):
                fields.append(f"{statistic:.3f}")
                fields.append(f"{pvalue:.3f}")
            print(",".join(fields), file=fh)


if __name__ == "__main__":
    main()
