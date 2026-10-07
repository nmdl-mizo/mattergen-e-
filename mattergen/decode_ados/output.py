"""Write the decoded DOS and its site metadata to a single ``.npz`` archive.

This module is dependency-light (standard library plus numpy) so it can be
unit-tested without importing pymatgen / torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .records import SiteRecord

ENERGIES_KEY = "energies"
METADATA_KEYS = ("structure_id", "site_index", "element", "label")


def build_arrays(
    dos: np.ndarray,
    energies: np.ndarray,
    records: Sequence[SiteRecord],
    dos_key: str,
) -> dict[str, np.ndarray]:
    """Assemble the archive contents, one array per key, all row-aligned."""
    if len(dos) != len(records):
        raise ValueError(f"Got {len(dos)} decoded rows for {len(records)} sites.")
    if dos.shape[1] != len(energies):
        raise ValueError(f"Got {dos.shape[1]} DOS bins for {len(energies)} energies.")
    if dos_key in {ENERGIES_KEY, *METADATA_KEYS}:
        raise ValueError(f"{dos_key!r} collides with a reserved archive key.")

    return {
        dos_key: dos,
        ENERGIES_KEY: energies,
        "structure_id": np.array([record.structure_id for record in records], dtype=np.str_),
        "site_index": np.array([record.site_index for record in records], dtype=np.int64),
        "element": np.array([record.element for record in records], dtype=np.str_),
        "label": np.array([record.label for record in records], dtype=np.str_),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Writing through a handle keeps the caller's filename: numpy appends a
    # ".npz" suffix when it is given a path that lacks one.
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
