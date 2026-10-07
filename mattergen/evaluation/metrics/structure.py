# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
from functools import cached_property
from typing import Sequence

import cachetools
import numpy as np
import numpy.typing
from pandas import DataFrame
from pymatgen.core.composition import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mattergen.evaluation.metrics.core import BaseMetricsCapability
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.dataset_matcher import (
    DisorderedDatasetUniquenessComputer,
    OrderedDatasetUniquenessComputer,
    get_dataset_matcher,
    matches_to_mask,
)
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
from mattergen.evaluation.utils.structure_matcher import (
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)
from mattergen.evaluation.utils.symmetry_analysis import (
    DefaultSpaceGroupAnalyzer,
    DisorderedSpaceGroupAnalyzer,
)


def get_space_group(
    structure: Structure,
    space_group_analyzer_cls: type[SpacegroupAnalyzer] = DefaultSpaceGroupAnalyzer,
) -> str:
    try:
        return space_group_analyzer_cls(structure=structure).get_space_group_symbol()
    except TypeError:
        # space group analysis failed, most likely due to overlapping atoms
        return "P1"


def all_structures_are_ordered(structures: Sequence[Structure]) -> bool:
    """Check if all structures are ordered."""
    return all([s.is_ordered for s in structures])


class StructureMetricsCapability(BaseMetricsCapability):
    name: str = "structure_capability"

    """Capability for computing structure metrics.
    The `structure_matcher` class determines how uniqueness and novelty are computed.
        atoms that could substitute for each other (via the Hume-Rothery rules) and then using the default pymatgen structure matching.
    """

    def __init__(
        self,
        structure_summaries: list[MetricsStructureSummary],
        reference_dataset: ReferenceDataset,
        structure_matcher: OrderedStructureMatcher
        | DisorderedStructureMatcher,  # how are uniqueness and novelty computed
        n_failed_jobs: int = 0,
    ) -> None:
        super().__init__(structure_summaries=structure_summaries, n_failed_jobs=n_failed_jobs)
        _structures = [s.structure for s in structure_summaries]
        all_structures_ordered = (
            all_structures_are_ordered(_structures) and reference_dataset.is_ordered
        )
        if not all_structures_ordered:
            assert isinstance(structure_matcher, DisorderedStructureMatcher), (
                "If at least one structure is disordered, "
                "structure_matcher must be a DisorderedStructureMatcher."
            )
            logger.info(
                "At least one structure is disordered. Using DisorderedDatasetUniquenessComputer."
            )
        self.reference_dataset = reference_dataset  # note: not all metrics use this, so it could be a separate capability
        self.structure_matcher = structure_matcher
        self.ensure_reference_dataset_has_material_ids()
        self.uniqueness_computer: OrderedDatasetUniquenessComputer | DisorderedDatasetUniquenessComputer = (
            OrderedDatasetUniquenessComputer(structure_matcher)
            if all_structures_ordered
            else DisorderedDatasetUniquenessComputer(structure_matcher)
        )
        self.dataset_matcher = get_dataset_matcher(all_structures_ordered, structure_matcher)

    def ensure_reference_dataset_has_material_ids(self) -> None:
        """
        We're using material_ids to match structures between the reference dataset and the data.
        If the reference dataset doesn't have material_ids, we add them here and set them
        to the index of the entry in the reference dataset.
        """
        if (
            len(self.reference_dataset) > 0
            and next(iter(self.reference_dataset)).data.get("material_id") is None
        ):
            logger.warning(
                "Reference dataset does not have material_ids. Adding material_ids to reference dataset."
            )
            for i, entry in enumerate(self.reference_dataset):
                if "material_id" in entry.data:
                    raise ValueError(
                        "Found material_id in some entries of the reference dataset, but not all."
                        "Please ensure that either all entries have material_ids or none do."
                    )
                entry.data["material_id"] = i

    @property
    def structures(self) -> list[Structure]:
        return [s.structure for s in self._structure_summaries]

    @cached_property
    def chemistry_agnostic_structures(self) -> list[Structure]:
        chemistry_agnostic_structures = [deepcopy(s) for s in self.structures]
        for s in chemistry_agnostic_structures:
            s.replace_species({Element(k.name): Element("Cs") for k in list(set(s.species))})

        return chemistry_agnostic_structures

    @cached_property
    def is_unique(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as `data_entries` in which each item is True
        for the first structure from a set of duplicates and otherwise False.
        """
        return self.uniqueness_computer(self.dataset)

    @cached_property
    def is_novel(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as `data_entries` in which each item is True
        for structures that are not present in the reference dataset and otherwise False.
        """
        novelty_mask = np.logical_not(self.is_in_reference)
        return novelty_mask

    @cached_property
    def is_in_reference(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as `data_entries` in which each item is True
        for structures that are present in the reference dataset and otherwise False.
        """

        return matches_to_mask(self.matches_in_reference.keys(), len(self.dataset))

    @cached_property
    def matches_in_reference(self) -> dict[int, list[str]]:
        return self.dataset_matcher(self.dataset, self.reference_dataset)

    @cached_property
    def is_explored(self) -> numpy.typing.NDArray[np.bool_]:
        """Returns a mask of whether structures are in explored chemical systems (>1 entry in reference)."""
        return np.array(
            [
                structure.composition.chemical_system in self.reference_dataset.entries_by_chemsys
                for structure in self.structures
            ]
        )

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            data={
                # including is_unique and is_novel here might trigger an expensive computation
                "is_unique": self.is_unique,
                "is_novel": self.is_novel,
                "is_explored": self.is_explored,
            },
            index=[e.entry_id for e in self.dataset],
        )

    @cached_property
    def num_atoms(self) -> numpy.typing.NDArray[np.int_]:
        return np.array([len(structure) for structure in self.structures])

    @cached_property
    def space_group_symbols(self) -> list[str]:
        return [get_space_group(structure) for structure in self.structures]

    @cached_property
    def chemistry_agnostic_space_group_symbols(self) -> list[str]:
        return [get_space_group(structure) for structure in self.chemistry_agnostic_structures]

    @cached_property
    def substitution_aware_space_group_symbols(self) -> list[str]:
        """
        Returns a list of space group symbols for each structure in the dataset, once the
        structures have been modified to account for possible substitutions of atoms that
        could substitute for each other (via the Hume-Rothery rules).
        """
        return [
            get_space_group(structure, DisorderedSpaceGroupAnalyzer)
            for structure in self.structures
        ]

    # Ignore "desc" for the cache because it is irrelevant.
    @cachetools.cached(cache={}, key=lambda self, *args, **kwargs: self)
    def compute_num_matches(
        self,
        desc: str = "",
    ) -> float:
        """
        Returns the number of matches between the data and reference entries.
        """
        num_matches = len(self.is_novel) - sum(self.is_novel)
        return num_matches
