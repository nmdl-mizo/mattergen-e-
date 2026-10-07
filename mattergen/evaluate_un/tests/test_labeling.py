"""Unit tests for the ``structure`` summary label (mattergen.evaluate_un.labeling)."""

import itertools

import pytest

from mattergen.evaluate_un.labeling import (
    NOVEL_LABEL,
    UNIQUE_LABEL,
    compute_structure_label,
    compute_structure_labels,
    is_unique_representative,
)

ALLOWED_LABELS = {"", "U", "N", "UN"}


@pytest.mark.parametrize(
    "in_reference, is_duplicated, representative_gen_id, gen_id, expected",
    [
        # in reference, duplicated and NOT the representative -> "" (neither flag)
        (True, True, "gen_0", "gen_1", ""),
        # in reference, not duplicated -> unique representative -> "U"
        (True, False, "gen_1", "gen_1", "U"),
        # in reference, duplicated but IS the representative -> "U"
        (True, True, "gen_1", "gen_1", "U"),
        # novel, duplicated and NOT the representative -> "N"
        (False, True, "gen_0", "gen_1", "N"),
        # novel, not duplicated -> unique representative + novel -> "UN"
        (False, False, "gen_1", "gen_1", "UN"),
        # novel, duplicated but IS the representative -> "UN"
        (False, True, "gen_1", "gen_1", "UN"),
    ],
)
def test_compute_structure_label_cases(
    in_reference, is_duplicated, representative_gen_id, gen_id, expected
):
    assert (
        compute_structure_label(
            in_reference=in_reference,
            is_duplicated=is_duplicated,
            representative_gen_id=representative_gen_id,
            gen_id=gen_id,
        )
        == expected
    )


def test_label_is_always_one_of_four_values():
    """Every combination of inputs must produce one of the four allowed labels."""
    rep_gen_pairs = [("gen_a", "gen_a"), ("gen_a", "gen_b")]  # match / mismatch
    for in_reference, is_duplicated, (rep, gen) in itertools.product(
        (True, False), (True, False), rep_gen_pairs
    ):
        label = compute_structure_label(
            in_reference=in_reference,
            is_duplicated=is_duplicated,
            representative_gen_id=rep,
            gen_id=gen,
        )
        assert label in ALLOWED_LABELS


def test_label_order_is_unique_then_novel():
    """When both flags are set the order is 'U' then 'N' (i.e. 'UN', never 'NU')."""
    label = compute_structure_label(
        in_reference=False,
        is_duplicated=False,
        representative_gen_id="g",
        gen_id="g",
    )
    assert label == UNIQUE_LABEL + NOVEL_LABEL == "UN"


def test_novel_flag_depends_only_on_in_reference():
    # in_reference=False always contributes the novel flag "N"...
    assert NOVEL_LABEL in compute_structure_label(
        in_reference=False, is_duplicated=True, representative_gen_id="a", gen_id="b"
    )
    # ...and in_reference=True never does.
    assert NOVEL_LABEL not in compute_structure_label(
        in_reference=True, is_duplicated=False, representative_gen_id="a", gen_id="a"
    )


@pytest.mark.parametrize(
    "is_duplicated, representative_gen_id, gen_id, expected",
    [
        (False, "gen_0", "gen_1", True),  # not duplicated -> representative
        (True, "gen_1", "gen_1", True),  # duplicated but IS the representative
        (True, "gen_0", "gen_1", False),  # duplicated and NOT the representative
    ],
)
def test_is_unique_representative(
    is_duplicated, representative_gen_id, gen_id, expected
):
    assert (
        is_unique_representative(
            is_duplicated=is_duplicated,
            representative_gen_id=representative_gen_id,
            gen_id=gen_id,
        )
        == expected
    )


def test_compute_structure_labels_matches_single_row_and_covers_all_values():
    results = [
        {"index": 0, "id": "gen_0", "in_reference": True},  # rep of {0, 1}
        {"index": 1, "id": "gen_1", "in_reference": True},  # non-rep dup
        {"index": 2, "id": "gen_2", "in_reference": False},  # singleton novel
        {"index": 3, "id": "gen_3", "in_reference": False},  # non-rep dup, novel
    ]
    representative_gen_id = {0: "gen_0", 1: "gen_0", 2: "gen_2", 3: "gen_4"}
    duplicate_gen_ids = {0: ["gen_1"], 1: ["gen_0"], 2: [], 3: ["gen_4"]}

    labels = compute_structure_labels(results, representative_gen_id, duplicate_gen_ids)

    assert labels == ["U", "", "UN", "N"]

    # The batch labels must agree with the single-row computation element-wise.
    for r in results:
        idx = r["index"]
        assert labels[idx] == compute_structure_label(
            in_reference=r["in_reference"],
            is_duplicated=bool(duplicate_gen_ids[idx]),
            representative_gen_id=representative_gen_id[idx],
            gen_id=r["id"],
        )


def test_compute_structure_labels_uses_fallbacks_for_missing_metadata():
    # No metadata at all: every structure is its own (non-duplicated) representative.
    results = [
        {"index": 0, "id": "gen_0", "in_reference": True},
        {"index": 1, "id": "gen_1", "in_reference": False},
    ]

    labels = compute_structure_labels(results, representative_gen_id={}, duplicate_gen_ids={})

    assert labels == ["U", "UN"]


def test_label_substring_selection_for_u_n_and_un_directories():
    """The U/N/UN directory selection is a substring test on the structure label."""
    labels = ["U", "", "UN", "N"]
    u_selected = [UNIQUE_LABEL in label for label in labels]
    n_selected = [NOVEL_LABEL in label for label in labels]
    un_selected = [UNIQUE_LABEL in label and NOVEL_LABEL in label for label in labels]
    assert u_selected == [True, False, True, False]  # "U" and "UN"
    assert n_selected == [False, False, True, True]  # "UN" and "N"
    assert un_selected == [False, False, True, False]  # only "UN"
