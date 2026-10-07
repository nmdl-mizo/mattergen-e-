"""Shared helpers for building batches of test data.

``SetEmbeddingType`` and friends call :meth:`ChemGraph.get_batch_size`, which
only works on a collated ``ChemGraphBatch``. Tests must therefore build their
inputs through :func:`mattergen.common.data.collate.collate` rather than
instantiating a bare ``ChemGraph``.
"""

from typing import Any, Sequence

import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate

# Carbon; the element is irrelevant to the property-embedding logic under test.
_DUMMY_ATOMIC_NUMBER = 6


def make_batch(
    num_atoms: Sequence[int],
    crystal_properties: dict[str, Sequence[float]] | None = None,
    extra_fields: dict[str, Sequence[float]] | None = None,
) -> ChemGraph:
    """Collate one structure per entry of ``num_atoms`` into a ``ChemGraphBatch``.

    Args:
        num_atoms: number of atoms of each structure in the batch.
        crystal_properties: name -> one value per *structure* (``len(num_atoms)`` values).
        extra_fields: per-structure values stored under a name that is not a
            registered property; used to test that unregistered fields are ignored.
    """
    crystal_properties = crystal_properties or {}
    extra_fields = extra_fields or {}

    for name, values in {**crystal_properties, **extra_fields}.items():
        assert len(values) == len(num_atoms), f"{name} needs one value per structure"

    graphs = []
    for structure_index, n in enumerate(num_atoms):
        # dict[str, Any]: the ChemGraph constructor declares tensor-subtype parameters
        # (LongTensor etc.) that plain tensor constructors do not satisfy for mypy.
        fields: dict[str, Any] = {
            name: torch.tensor([values[structure_index]], dtype=torch.float)
            for name, values in {**crystal_properties, **extra_fields}.items()
        }
        graphs.append(
            ChemGraph(
                pos=torch.rand(n, 3),
                cell=torch.eye(3).unsqueeze(0),
                atomic_numbers=torch.full((n,), _DUMMY_ATOMIC_NUMBER, dtype=torch.long),  # type: ignore[arg-type]
                num_atoms=torch.tensor([n]),
                **fields,
            )
        )
    return collate(graphs)


def values_with_nan_at(indices: Sequence[int], length: int) -> list[float]:
    """A list of ``length`` finite values, with ``torch.nan`` at ``indices``."""
    values = [float(i) for i in range(length)]
    for i in indices:
        values[i] = float("nan")
    return values
