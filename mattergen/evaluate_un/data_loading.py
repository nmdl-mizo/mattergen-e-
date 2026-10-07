"""Loading the reference dataset and the generated CIF structures.

These helpers depend on pymatgen / mattergen and are therefore slow to import;
they are only used by the real pipeline, not by the light-weight unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pymatgen.io.cif import CifParser

from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary


def load_reference_dataset(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Reference dataset not found: {path}")
    return LMDBGZSerializer().deserialize(str(path))


def find_cif_files(structures_dir: Path) -> list[Path]:
    if not structures_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {structures_dir}")
    return sorted(structures_dir.rglob("*.cif"))


def load_structures(cif_files: Iterable[Path]):
    structures = []
    for cif_path in cif_files:
        parser = CifParser(str(cif_path))
        structures.append(parser.parse_structures(primitive=False)[0])
    return structures


def build_summaries(structures):
    return [MetricsStructureSummary.from_structure(s) for s in structures]


def get_gen_id(cif_path: Path) -> str:
    """Extract generated structure ID (e.g. gen_1004) from CIF path."""
    return Path(cif_path).stem
