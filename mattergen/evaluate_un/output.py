"""Write evaluation results to CSV and copy selected CIF files.

This module is dependency-light (standard library only) so it can be unit-tested
without importing pymatgen / mattergen. The ``mask`` passed to
:func:`copy_cifs_by_mask` only needs to support boolean indexing, so plain Python
lists work as well as numpy arrays.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .labeling import compute_structure_label

# Column order of the output CSV. ``structure`` is a derived summary column (see
# ``mattergen.evaluate_un.labeling``) appended at the end so existing column
# positions are preserved for any downstream consumer.
CSV_FIELDNAMES = [
    "index",
    "id",
    "cif_path",
    "in_reference",
    "matched_material_ids",
    "is_duplicated",
    "representative_gen_id",
    "duplicate_gen_ids",
    "structure",
]


def save_csv(
    results: list[dict],
    path: Path,
    representative_gen_id: dict[int, str],
    duplicate_gen_ids: dict[int, list[str]],
) -> None:
    """Write per-structure membership results (with duplicate metadata) to CSV.

    In addition to the raw membership/duplicate fields, each row gets a derived
    ``structure`` label (one of ``""``, ``"U"``, ``"N"``, ``"UN"``); see
    :func:`mattergen.evaluate_un.labeling.compute_structure_label`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            idx = r["index"]
            row = dict(r)
            row["matched_material_ids"] = ";".join(row["matched_material_ids"])
            dup_ids = duplicate_gen_ids.get(idx, [])
            row["is_duplicated"] = bool(dup_ids)
            row["representative_gen_id"] = representative_gen_id.get(idx, row["id"])
            row["duplicate_gen_ids"] = ";".join(dup_ids)
            row["structure"] = compute_structure_label(
                in_reference=bool(row["in_reference"]),
                is_duplicated=bool(row["is_duplicated"]),
                representative_gen_id=row["representative_gen_id"],
                gen_id=row["id"],
            )
            writer.writerow(row)


def copy_cifs_by_mask(
    cif_files: list[Path],
    mask,
    output_dir: Path,
) -> int:
    """Copy ``cif_files[idx]`` into ``output_dir`` for every truthy ``mask[idx]``.

    Any ``*.cif`` files already present in ``output_dir`` (e.g. left over from a
    previous run) are deleted first, so that on return the directory contains
    exactly the CIFs selected by ``mask`` for the current run. Only ``*.cif`` files
    are removed; any other files in the directory are left untouched.

    ``mask`` is positionally aligned with ``cif_files`` and only needs to support
    boolean indexing (a plain ``list[bool]`` or a numpy array both work).

    Returns the number of files copied.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove pre-existing CIF files so stale structures from a previous run do not
    # linger alongside the freshly selected ones.
    for stale_cif in output_dir.glob("*.cif"):
        if stale_cif.is_file():
            stale_cif.unlink()
    count = 0
    for idx, cif_path in enumerate(cif_files):
        if bool(mask[idx]):
            shutil.copy2(cif_path, output_dir / cif_path.name)
            count += 1
    return count
