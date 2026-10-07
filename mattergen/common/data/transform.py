# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from typing import List, Protocol, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from pymatgen.core import Composition

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.data_utils import (
    compute_lattice_polar_decomposition,
    get_element_symbol,
)
from mattergen.common.utils.globals import MAX_ATOMIC_NUM


class Transform(Protocol):
    def __call__(self, sample: ChemGraph) -> ChemGraph:
        ...

def symmetrize_lattice(sample: ChemGraph) -> ChemGraph:
    return sample.replace(cell=compute_lattice_polar_decomposition(sample.cell))


def set_chemical_system(sample: ChemGraph) -> ChemGraph:
    chemsys = (
        torch.eye(MAX_ATOMIC_NUM + 1, device=sample.atomic_numbers.device)[
            sample.atomic_numbers
        ].sum(0)
        > 0
    ).float()[None]
    return sample.replace(chemical_system=chemsys)


def set_chemical_system_string(sample: ChemGraph) -> ChemGraph:
    return sample.replace(
        chemical_system=Composition(
            {get_element_symbol(Z=i.item()): 1 for i in sample.atomic_numbers}
        ).chemical_system
    )


class BaseSiteStructureTransform(Transform):
    def __init__(self, structure_name: str, shape: List[int] | int | None = None):
        self.structure_name = structure_name
        self.shape = shape if isinstance(shape, list) else (shape,) if shape is not None else ()

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        n_atoms = sample.atomic_numbers.shape[0]
        return sample.replace(
            **{
                self.structure_name: torch.full(
                    (n_atoms, *self.shape), float("nan"), dtype=torch.float32
                )
            }
        )


def set_bader_charge(sample: ChemGraph) -> ChemGraph:
    if "bader_charge" in sample:
        return sample
    transform = BaseSiteStructureTransform(structure_name="bader_charge", shape=None)
    return transform(sample)


def set_atomic_dos(sample: ChemGraph) -> ChemGraph:
    if "atomic_dos" in sample:
        return sample
    transform = BaseSiteStructureTransform(structure_name="atomic_dos", shape=30)
    return transform(sample)


class SetProperty:
    def __init__(self, property_name: str, value: float | Sequence[str]):
        self.property_name = property_name
        self.value = (
            torch.tensor(value, dtype=torch.float)
            if isinstance(value, float) or isinstance(value, int)
            else value
        )

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        return sample.replace(**{self.property_name: self.value})


class SetSitePropertyFromDistribution(Protocol):
    def __call__(self, samples: list[dict[str, float]], *args, **kwargs) -> NDArray: ...


class SetIntercalantOriginProperty(SetSitePropertyFromDistribution):
    def __init__(self):
        super().__init__()

    def __call__(self, samples: list[dict[str, float]], num_samples: int) -> NDArray:
        assert len(samples) == num_samples, "Number of samples must match the number of samples in the distribution"
        assert all(("intercalant" in sample) and (sample["intercalant"] <= sample["n"]) for sample in samples), "intercalant must be less than or equal to n"

        intercalant_origin = np.concatenate(
            [
                [1] * int(sample["intercalant"])
                + [0] * (int(sample["n"]) - int(sample["intercalant"]))
                for sample in samples
            ],
            axis=-1,
        )
        return intercalant_origin
