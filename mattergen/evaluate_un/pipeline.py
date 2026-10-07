"""End-to-end orchestration of the structure evaluation pipeline."""

from __future__ import annotations

import argparse

from .cli import parse_args
from .data_loading import (
    build_summaries,
    find_cif_files,
    load_reference_dataset,
    load_structures,
)
from .duplicates import build_duplicate_groups
from .labeling import NOVEL_LABEL, UNIQUE_LABEL, compute_structure_labels
from .membership import evaluate_membership
from .metadata import build_duplicate_metadata
from .output import copy_cifs_by_mask, save_csv


def run(args: argparse.Namespace | None = None) -> int:
    """Run the full evaluation pipeline and return a process exit code."""
    if args is None:
        args = parse_args()

    ref_dataset = load_reference_dataset(args.reference_dataset)
    cif_files = find_cif_files(args.structures_dir)
    if not cif_files:
        raise RuntimeError(f"No CIF files found under: {args.structures_dir}")

    structures = load_structures(cif_files)
    summaries = build_summaries(structures)
    # `evaluate_membership` also returns the capability's novelty mask, but the
    # N selection below is derived from `in_reference` via the structure label
    # (these are equivalent, since is_novel == not is_in_reference) so the output
    # directories stay consistent with the CSV; hence the third value is ignored.
    results, is_unique_mask, _ = evaluate_membership(summaries, ref_dataset, cif_files)

    # Build duplicate metadata (based on gen_id)
    groups = build_duplicate_groups(structures, is_unique_mask)
    representative_gen_id, duplicate_gen_ids = build_duplicate_metadata(results, groups)

    save_csv(
        results=results,
        path=args.output_csv,
        representative_gen_id=representative_gen_id,
        duplicate_gen_ids=duplicate_gen_ids,
    )

    # Select CIFs for the U / N / UN output directories from the same structure
    # labels written to the CSV: the U directory receives every structure whose
    # label contains "U" ("U" or "UN"); the N directory every label containing
    # "N" ("N" or "UN"); the UN directory only structures that are both unique
    # and novel (label "UN"). A "UN" structure is therefore copied into all three
    # directories. Each directory is optional and only written when its
    # corresponding --*_output_dir flag is provided.
    labels = compute_structure_labels(results, representative_gen_id, duplicate_gen_ids)

    copied_u = copied_n = copied_un = 0
    if args.u_output_dir is not None:
        copied_u = copy_cifs_by_mask(
            cif_files, [UNIQUE_LABEL in label for label in labels], args.u_output_dir
        )
    if args.n_output_dir is not None:
        copied_n = copy_cifs_by_mask(
            cif_files, [NOVEL_LABEL in label for label in labels], args.n_output_dir
        )
    if args.un_output_dir is not None:
        copied_un = copy_cifs_by_mask(
            cif_files,
            [UNIQUE_LABEL in label and NOVEL_LABEL in label for label in labels],
            args.un_output_dir,
        )

    print(f"Checked {len(results)} structures.")
    print(f"CSV written to : {args.output_csv}")
    if args.u_output_dir is not None:
        print(f"Copied {copied_u} unique (U) CIFs to: {args.u_output_dir}")
    if args.n_output_dir is not None:
        print(f"Copied {copied_n} novel (N) CIFs to: {args.n_output_dir}")
    if args.un_output_dir is not None:
        print(f"Copied {copied_un} unique+novel (UN) CIFs to: {args.un_output_dir}")
    return 0
