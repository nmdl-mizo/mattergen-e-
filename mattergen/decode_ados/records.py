"""Identity of a single decoded atom.

Kept in its own standard-library-only module so that both the CIF reader and the
archive writer can depend on it without dragging in pymatgen / torch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteRecord:
    structure_id: str
    site_index: int
    element: str
    label: str
