"""Read latent vectors and site metadata from a directory of CIF files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

from .cif_site_loop import parse_atom_site_loop
from .records import SiteRecord

LABEL_COLUMN = "label"


class LatentReadError(ValueError):
    """Raised when a CIF file does not carry a usable latent vector."""


def find_cif_files(structures_dir: Path) -> list[Path]:
    if not structures_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {structures_dir}")
    return sorted(structures_dir.rglob("*.cif"))


def check_unique_structure_ids(cif_files: Sequence[Path]) -> None:
    """Reject stem collisions across subdirectories.

    Rows of the output archive are identified by the CIF stem alone, so two
    files sharing one would produce indistinguishable entries.
    """
    seen: dict[str, Path] = {}
    for cif_path in cif_files:
        structure_id = cif_path.stem
        if structure_id in seen:
            raise LatentReadError(
                f"Duplicate structure id {structure_id!r}: {seen[structure_id]} and {cif_path}"
            )
        seen[structure_id] = cif_path


def latent_column_names(property_prefix: str, latent_dim: int) -> list[str]:
    return [f"{property_prefix}_{i}" for i in range(latent_dim)]


def read_latents(
    cif_files: Iterable[Path],
    property_prefix: str,
    latent_dim: int,
) -> tuple[np.ndarray, list[SiteRecord]]:
    """Return the per-atom latent vectors and their site records.

    Rows follow the site order of the parsed ``Structure``, which pymatgen does
    not keep aligned with the CIF row order, so values are joined on
    ``_atom_site_label`` rather than by position.
    """
    column_names = latent_column_names(property_prefix, latent_dim)
    latents: list[list[float]] = []
    records: list[SiteRecord] = []

    for cif_path in cif_files:
        content = cif_path.read_text(encoding="utf-8")
        structure = _parse_structure(cif_path, content)
        columns = _read_columns(cif_path, content, column_names)
        row_of_label = _index_rows_by_label(cif_path, columns[LABEL_COLUMN], len(structure))

        for site_index, site in enumerate(structure):
            row = row_of_label[site.label]
            latents.append(
                [
                    _parse_value(cif_path, site.label, name, columns[name][row])
                    for name in column_names
                ]
            )
            records.append(
                SiteRecord(
                    structure_id=cif_path.stem,
                    site_index=site_index,
                    element=site.specie.symbol,
                    label=site.label,
                )
            )

    if not latents:
        raise LatentReadError("No sites were read from the given CIF files.")
    return np.asarray(latents, dtype=np.float32), records


def _parse_structure(cif_path: Path, content: str) -> Structure:
    structures = CifParser.from_str(content).parse_structures(primitive=False)
    if not structures:
        raise LatentReadError(f"No structure could be parsed from {cif_path}")
    return structures[0]


def _read_columns(cif_path: Path, content: str, column_names: Sequence[str]) -> dict[str, list[str]]:
    columns = parse_atom_site_loop(content)
    if LABEL_COLUMN not in columns:
        raise LatentReadError(f"{cif_path} has no _atom_site_label column.")
    missing = [name for name in column_names if name not in columns]
    if missing:
        raise LatentReadError(
            f"{cif_path} is missing {len(missing)} of the {len(column_names)} latent "
            f"columns, starting with {missing[0]}."
        )
    return columns


def _index_rows_by_label(cif_path: Path, labels: Sequence[str], n_sites: int) -> dict[str, int]:
    if len(labels) != n_sites:
        raise LatentReadError(
            f"{cif_path} has {len(labels)} atom site rows but {n_sites} parsed sites."
        )
    row_of_label = {label: row for row, label in enumerate(labels)}
    if len(row_of_label) != len(labels):
        raise LatentReadError(f"{cif_path} has duplicate _atom_site_label values.")
    return row_of_label


def _parse_value(cif_path: Path, label: str, column: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise LatentReadError(
            f"{cif_path}, site {label}, column {column}: {raw!r} is not a number."
        ) from exc
    if not math.isfinite(value):
        raise LatentReadError(f"{cif_path}, site {label}, column {column}: value is {value}.")
    return value
