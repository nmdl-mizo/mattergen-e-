# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from copy import deepcopy
from functools import cached_property

from pandas import DataFrame

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary


class BaseMetricsCapability:
    """Base class for capabilities."""

    name: str = "base_capability"

    def __init__(
        self, structure_summaries: list[MetricsStructureSummary], n_failed_jobs: int = 0
    ) -> None:
        assert len(structure_summaries) > 0, "No data provided."
        self._structure_summaries = structure_summaries
        self.n_failed_jobs = n_failed_jobs

    @property
    def total_submitted_jobs(self) -> int:
        return len(self.dataset) + self.n_failed_jobs

    @cached_property
    def dataset(self) -> ReferenceDataset:
        """
        Returns a ReferenceDataset. While not all capabilities require energies,
        the entry IDs are useful to keep track of entry IDs.
        """
        data_entries = [deepcopy(s.entry) for s in self._structure_summaries]
        for i, e in enumerate(data_entries):
            e.entry_id = i

        return ReferenceDataset.from_entries("data_entries", data_entries)

    @abc.abstractmethod
    def as_dataframe(self) -> DataFrame:
        """Returns a pandas DataFrame containing information about this capability."""
