"""Derive per-structure duplicate metadata from duplicate groups.

This module is intentionally dependency-light (standard library only) so it can
be imported and unit-tested without pulling in pymatgen / mattergen.
"""

from __future__ import annotations


def build_duplicate_metadata(
    results: list[dict],
    groups: dict[int, list[int]],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """From duplicate groups, derive per-index representative and duplicate gen_ids.

    Args:
        results: per-structure result dicts (must contain ``id``), indexed by
            position so ``results[idx]`` corresponds to structure index ``idx``.
        groups: mapping from representative index to the list of member indices
            (including the representative itself).

    Returns:
        representative_gen_id: idx -> representative id
        duplicate_gen_ids: idx -> list of duplicate gen_ids (other members)
    """
    representative_gen_id: dict[int, str] = {}
    duplicate_gen_ids: dict[int, list[str]] = {}

    # First, fill in data for indices that belong to a group
    for rep_idx, members in groups.items():
        rep_gen_id = results[rep_idx]["id"]
        for idx in members:
            representative_gen_id[idx] = rep_gen_id
        for idx in members:
            others = [m for m in members if m != idx]
            duplicate_gen_ids[idx] = [results[m]["id"] for m in others]

    # Ensure all indices are covered, including those not in any group
    for idx in range(len(results)):
        if idx not in representative_gen_id:
            representative_gen_id[idx] = results[idx]["id"]
        if idx not in duplicate_gen_ids:
            duplicate_gen_ids[idx] = []

    return representative_gen_id, duplicate_gen_ids
