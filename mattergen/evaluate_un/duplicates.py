"""Group duplicate generated structures (used to derive uniqueness metadata).

We intentionally group by chemical system (chemsys) to reduce comparisons while
preserving correctness of ``DefaultDisorderedStructureMatcher``, which can match
disordered structures using fractional composition checks. The per-chemsys work
is embarrassingly parallel, so it is farmed out to a multiprocessing pool when
there is more than one bucket.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher

# ---------------------------------------------------------------------------
# Module-level helpers for multiprocessing (must be picklable)
# ---------------------------------------------------------------------------

_pool_structures: list = []
_pool_is_unique: np.ndarray = np.array([], dtype=bool)


def _init_pool_worker(structures_ref: list, is_unique_ref: np.ndarray) -> None:
    """Initializer for Pool workers — stores shared data in module globals."""
    global _pool_structures, _pool_is_unique
    _pool_structures = structures_ref
    _pool_is_unique = is_unique_ref


def _find_groups_in_chemsys(
    item: tuple[str, list[int]],
    structures: list,
    is_unique: np.ndarray,
) -> dict[int, list[int]]:
    """Find duplicate groups within a single chemsys bucket (explicit args)."""
    _chemsys, idxs = item
    matcher = DefaultDisorderedStructureMatcher()

    reps = [i for i in idxs if bool(is_unique[i])]
    local_groups: dict[int, list[int]] = {rep: [rep] for rep in reps}
    assigned: set[int] = set()

    for rep in reps:
        if rep in assigned:
            continue
        assigned.add(rep)
        rep_struct = structures[rep]
        for idx in idxs:
            if idx == rep or idx in assigned:
                continue
            if matcher.fit(rep_struct, structures[idx]):
                local_groups[rep].append(idx)
                assigned.add(idx)

    return local_groups


def _find_groups_in_chemsys_pooled(
    item: tuple[str, list[int]],
) -> dict[int, list[int]]:
    """Wrapper for Pool workers — reads shared data from module globals."""
    return _find_groups_in_chemsys(item, _pool_structures, _pool_is_unique)


def _get_num_workers() -> int:
    # Respect SLURM allocation if present.
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 1))


def build_duplicate_groups(
    structures,
    is_unique: np.ndarray,
) -> dict[int, list[int]]:
    """Build duplicate groups among generated structures.

    Returns:
        Mapping from representative index to list of member indices
        (including the representative itself).
    """
    # Group indices by chemsys (safe pre-filter for DisorderedStructureMatcher).
    indices_by_chemsys: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(structures):
        indices_by_chemsys[s.composition.chemical_system].append(idx)

    items = list(indices_by_chemsys.items())
    all_groups: dict[int, list[int]] = {}

    if len(items) <= 1:
        for item in tqdm(items, total=len(items), desc="Finding duplicate groups (chemsys)"):
            all_groups.update(_find_groups_in_chemsys(item, structures, is_unique))
        return all_groups

    from multiprocessing import Pool

    n_workers = _get_num_workers()
    with Pool(
        processes=n_workers,
        initializer=_init_pool_worker,
        initargs=(structures, is_unique),
    ) as pool:
        for local_groups in tqdm(
            pool.imap_unordered(_find_groups_in_chemsys_pooled, items),
            total=len(items),
            desc=f"Finding duplicate groups (chemsys, workers={n_workers})",
        ):
            all_groups.update(local_groups)

    return all_groups
