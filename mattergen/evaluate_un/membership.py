"""Evaluate membership of generated structures against the reference dataset.

Produces, per generated structure, whether it is contained in the reference
dataset along with the uniqueness / novelty masks computed by MatterGen.
"""

from __future__ import annotations

import numpy as np

from mattergen.evaluation.metrics.structure import StructureMetricsCapability
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher

from .data_loading import get_gen_id


def evaluate_membership(
    summaries: list[MetricsStructureSummary],
    reference_dataset,
    cif_files: list,
):
    matcher = DefaultDisorderedStructureMatcher()
    capability = StructureMetricsCapability(
        structure_summaries=summaries,
        reference_dataset=reference_dataset,
        structure_matcher=matcher,
    )

    in_ref_mask: np.ndarray = capability.is_in_reference
    is_unique_mask: np.ndarray = capability.is_unique
    is_novel_mask: np.ndarray = capability.is_novel
    matches = capability.matches_in_reference

    results: list[dict] = []
    for idx, cif_path in enumerate(cif_files):
        gen_id = get_gen_id(cif_path)
        results.append(
            {
                "index": idx,
                "id": gen_id,
                "cif_path": str(cif_path),
                "in_reference": bool(in_ref_mask[idx]),
                "matched_material_ids": matches.get(idx, []),
            }
        )
    return results, is_unique_mask, is_novel_mask
