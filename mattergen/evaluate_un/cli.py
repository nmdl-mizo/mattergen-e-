"""Command-line argument parsing for the evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether generated CIF structures are contained in a reference "
            "dataset (database) using MatterGen's evaluation utilities."
        )
    )
    parser.add_argument(
        "--structures_dir",
        type=Path,
        required=True,
        help="Directory containing generated CIF files (searched recursively).",
    )
    parser.add_argument(
        "--reference_dataset",
        type=Path,
        required=True,
        help="Path to reference dataset in LMDBGZSerializer gzip format (e.g. *.gz).",
    )
    parser.add_argument(
        "--u_output_dir",
        type=Path,
        default=None,
        help=(
            "If set, copy CIFs that are unique representatives (structure label "
            "'U' or 'UN', i.e. not a non-representative duplicate) into this "
            "directory."
        ),
    )
    parser.add_argument(
        "--n_output_dir",
        type=Path,
        default=None,
        help=(
            "If set, copy CIFs that are novel w.r.t. the reference dataset "
            "(structure label 'N' or 'UN') into this directory."
        ),
    )
    parser.add_argument(
        "--un_output_dir",
        type=Path,
        default=None,
        help=(
            "If set, copy CIFs that are both unique representatives *and* novel "
            "(structure label 'UN') into this directory."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("generated_vs_db.csv"),
        help="Output CSV file path for membership results.",
    )
    return parser.parse_args()
