# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch
from torch_scatter import scatter_add

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.transform import set_chemical_system_string
from mattergen.common.utils.globals import MAX_ATOMIC_NUM, PROPERTY_SOURCE_IDS
from mattergen.denoiser import mask_disallowed_elements
from mattergen.property_embeddings import (
    ChemicalSystemMultiHotEmbedding,
    SetConditionalEmbeddingType,
    SetEmbeddingType,
    SetUnconditionalEmbeddingType,
    get_cond_field_names_in_batch,
    get_use_unconditional_embedding,
    replace_use_unconditional_embedding,
)
from mattergen.tests.conftest import make_batch, values_with_nan_at

# Number of atoms of each structure in the batches below: 6 structures, 15 atoms.
# The sizes are deliberately non-uniform so that a per-structure mask can never be
# confused with a per-atom one.
NUM_ATOMS = [2, 3, 1, 4, 2, 3]
NUM_STRUCTURES = len(NUM_ATOMS)

# The property names are read from the registry rather than hard-coded: a property is
# only treated as a conditional field if it is registered there (see
# ``test_unregistered_field_is_not_a_conditional_field``), so hard-coded names silently
# turn these tests into no-ops when the registry changes.
requires_two_properties = pytest.mark.skipif(
    len(PROPERTY_SOURCE_IDS) < 2,
    reason=f"needs two registered properties, PROPERTY_SOURCE_IDS={PROPERTY_SOURCE_IDS}",
)
PROPERTY_A = PROPERTY_SOURCE_IDS[0]
PROPERTY_B = PROPERTY_SOURCE_IDS[1] if len(PROPERTY_SOURCE_IDS) > 1 else PROPERTY_SOURCE_IDS[0]

# Structures whose property value is missing (NaN), i.e. structures that must fall back
# to the unconditional embedding even when p_unconditional=0.
NAN_STRUCTURES_A = [0, 2, 3]
NAN_STRUCTURES_B = [1, 2, 4]
# With dropout_fields_iid=False a structure uses the conditional embedding only if
# *every* conditional field is present, so the masked structures are the union.
NAN_STRUCTURES_EITHER = sorted(set(NAN_STRUCTURES_A) | set(NAN_STRUCTURES_B))


@pytest.mark.parametrize("p_unconditional", [0.0, 1.0])
def test_pre_corruption_fn(p_unconditional: float):
    """Structures with a missing (NaN) property value use the unconditional embedding.

    With p_unconditional=0 only those structures are masked; with p_unconditional=1
    every structure is masked regardless of whether its label is present.
    """
    batch = make_batch(
        num_atoms=NUM_ATOMS,
        crystal_properties={
            PROPERTY_A: values_with_nan_at(NAN_STRUCTURES_A, NUM_STRUCTURES),
        },
    )

    x_with_mask = SetEmbeddingType(p_unconditional=p_unconditional, dropout_fields_iid=False)(
        x=batch
    )
    mask = get_use_unconditional_embedding(batch=x_with_mask, cond_field=PROPERTY_A)

    assert mask.shape == (NUM_STRUCTURES, 1)
    expected_masked = {0.0: NAN_STRUCTURES_A, 1.0: list(range(NUM_STRUCTURES))}[p_unconditional]
    assert mask.squeeze(-1).nonzero().flatten().tolist() == expected_masked


@requires_two_properties
@pytest.mark.parametrize(
    "p_unconditional, dropout_fields_iid", [(0.0, True), (0.0, False), (1.0, True), (1.0, False)]
)
def test_pre_corruption_fn_multi(p_unconditional: float, dropout_fields_iid: bool):
    """With two conditional fields, ``dropout_fields_iid`` decides how they interact.

    When True each field is masked according to its own missing values; when False a
    structure keeps its conditional embedding only if all fields are present.
    """
    batch = make_batch(
        num_atoms=NUM_ATOMS,
        crystal_properties={
            PROPERTY_A: values_with_nan_at(NAN_STRUCTURES_A, NUM_STRUCTURES),
            PROPERTY_B: values_with_nan_at(NAN_STRUCTURES_B, NUM_STRUCTURES),
        },
    )

    x_with_mask = SetEmbeddingType(
        p_unconditional=p_unconditional, dropout_fields_iid=dropout_fields_iid
    )(x=batch)

    for cond_field, nan_structures in [
        (PROPERTY_A, NAN_STRUCTURES_A),
        (PROPERTY_B, NAN_STRUCTURES_B),
    ]:
        mask = get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field)
        assert mask.shape == (NUM_STRUCTURES, 1)

        if p_unconditional == 1.0:
            expected_masked = list(range(NUM_STRUCTURES))
        elif dropout_fields_iid:
            expected_masked = nan_structures
        else:
            expected_masked = NAN_STRUCTURES_EITHER

        assert mask.squeeze(-1).nonzero().flatten().tolist() == expected_masked, cond_field


@pytest.mark.parametrize("dropout_fields_iid", [True, False])
def test_pre_corruption_fn_without_conditional_fields_is_a_no_op(dropout_fields_iid: bool):
    """A batch carrying no registered property is returned unchanged."""
    batch = make_batch(num_atoms=NUM_ATOMS)

    x_with_mask = SetEmbeddingType(p_unconditional=0.0, dropout_fields_iid=dropout_fields_iid)(
        x=batch
    )

    assert get_cond_field_names_in_batch(x=batch) == []
    assert x_with_mask is batch


@requires_two_properties
def test_remove_conditioning_fn():
    """``SetUnconditionalEmbeddingType`` masks every conditional field of every structure."""
    batch = make_batch(
        num_atoms=NUM_ATOMS,
        crystal_properties={
            PROPERTY_A: values_with_nan_at([], NUM_STRUCTURES),
            PROPERTY_B: values_with_nan_at([], NUM_STRUCTURES),
        },
    )
    # Guard against the test silently passing through the "field not found" fallback of
    # get_use_unconditional_embedding, which also returns all-True.
    assert sorted(get_cond_field_names_in_batch(x=batch)) == sorted([PROPERTY_A, PROPERTY_B])

    x_with_mask = SetUnconditionalEmbeddingType()(x=batch)

    for cond_field in [PROPERTY_A, PROPERTY_B]:
        torch.testing.assert_close(
            get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field),
            torch.ones((NUM_STRUCTURES, 1), dtype=torch.bool),
        )


@requires_two_properties
def test_keep_conditioning_fn():
    """``SetConditionalEmbeddingType`` keeps the conditional embedding for every field."""
    batch = make_batch(
        num_atoms=NUM_ATOMS,
        crystal_properties={
            PROPERTY_A: values_with_nan_at([], NUM_STRUCTURES),
            PROPERTY_B: values_with_nan_at([], NUM_STRUCTURES),
        },
    )

    x_with_mask = SetConditionalEmbeddingType()(x=batch)

    for cond_field in [PROPERTY_A, PROPERTY_B]:
        torch.testing.assert_close(
            get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field),
            torch.zeros((NUM_STRUCTURES, 1), dtype=torch.bool),
        )


def test_unregistered_field_is_not_a_conditional_field():
    """Only fields registered in ``PROPERTY_SOURCE_IDS`` are conditioned on.

    A value stored under an unregistered name is ignored by ``SetEmbeddingType``, and
    querying it falls back to "use the unconditional embedding". This is why a test must
    take its property names from the registry: renaming or dropping an entry there turns
    conditioning off silently rather than raising.
    """
    unregistered = "not_a_registered_property"
    assert unregistered not in PROPERTY_SOURCE_IDS

    batch = make_batch(
        num_atoms=NUM_ATOMS,
        extra_fields={unregistered: values_with_nan_at([], NUM_STRUCTURES)},
    )

    assert get_cond_field_names_in_batch(x=batch) == []

    x_with_mask = SetConditionalEmbeddingType()(x=batch)
    torch.testing.assert_close(
        get_use_unconditional_embedding(batch=x_with_mask, cond_field=unregistered),
        torch.ones((NUM_STRUCTURES, 1), dtype=torch.bool),
    )


@pytest.mark.parametrize("zero_based_predictions", [True, False])
def test_mask_disallowed_elements(zero_based_predictions: bool):
    torch.manual_seed(23232)
    samples = [
        ChemGraph(
            pos=torch.rand(10, 3),
            num_atoms=torch.tensor([10]),
            atomic_numbers=6 * torch.ones((10,), dtype=torch.int),
            cell=torch.eye(3),
        ),
        ChemGraph(
            pos=torch.rand(5, 3),
            num_atoms=torch.tensor([5]),
            # La, Na, O, Sb, Sc
            atomic_numbers=torch.tensor([57, 11, 8, 51, 21]),
            cell=torch.eye(3),
        ),
        ChemGraph(
            pos=torch.rand(15, 3),
            num_atoms=torch.tensor([15]),
            # La, Na, O, Sb, Sc
            atomic_numbers=torch.tensor([57, 11, 8, 51, 21, 57, 11, 8, 51, 21, 57, 11, 8, 51, 21]),
            cell=torch.eye(3),
        ),
    ]

    transform = set_chemical_system_string
    batch = collate([transform(sample) for sample in samples])
    assert hasattr(batch, "chemical_system")  # mypy
    assert hasattr(batch, "pos")  # mypy
    assert hasattr(batch, "batch")  # mypy
    assert hasattr(batch, "cell")  # mypy
    assert hasattr(batch, "atomic_numbers")  # mypy
    assert hasattr(batch, "num_atoms")  # mypy
    mask = torch.tensor([0, 0, 1], dtype=torch.bool)[:, None]

    batch_chemgraph = ChemGraph(
        pos=batch.pos,
        cell=batch.cell,
        atomic_numbers=batch.atomic_numbers,
        num_atoms=batch.num_atoms,
        chemical_system=batch.chemical_system,
    )
    batch_chemgraph = replace_use_unconditional_embedding(batch=batch_chemgraph, use_unconditional_embedding={"chemical_system": mask})  # type: ignore

    example_logits = torch.randn(batch.pos.shape[0], MAX_ATOMIC_NUM + 1)
    masked_logits = mask_disallowed_elements(
        logits=example_logits,
        x=batch_chemgraph,
        batch_idx=batch.batch,
        predictions_are_zero_based=zero_based_predictions,
    )
    sampled = torch.distributions.Categorical(logits=masked_logits).sample() + int(
        zero_based_predictions
    )
    sampled_onehot = torch.eye(MAX_ATOMIC_NUM + 1)[sampled]
    sampled_chemical_systems = scatter_add(sampled_onehot, batch.batch, dim=0)

    # shape=(Nbatch, MAX_NUM_ATOMS+1)
    chemsys_multi_hot: torch.LongTensor = ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(
        x=ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(x=batch.chemical_system),
        device=mask.device,
    )

    for ix, system in enumerate(sampled_chemical_systems):
        sampled_types = system.nonzero()[:, 0].tolist()

        chemsys = chemsys_multi_hot[ix].nonzero()[:, 0].tolist()

        if mask[ix] == 0:
            assert set(sampled_types).difference(set(chemsys)) == set()
        else:
            assert set(sampled_types).difference(set(chemsys)) != set()
