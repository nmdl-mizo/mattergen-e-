"""Unit tests for per-structure duplicate metadata (mattergen.evaluate_un.metadata)."""

from mattergen.evaluate_un.metadata import build_duplicate_metadata


def test_groups_and_fallback_for_ungrouped_indices():
    results = [
        {"index": 0, "id": "gen_0"},
        {"index": 1, "id": "gen_1"},
        {"index": 2, "id": "gen_2"},
        {"index": 3, "id": "gen_3"},
    ]
    # index 0 represents the duplicate group {0, 2}; index 1 is its own singleton;
    # index 3 is intentionally absent from `groups` to exercise the fallback path.
    groups = {0: [0, 2], 1: [1]}

    representative_gen_id, duplicate_gen_ids = build_duplicate_metadata(results, groups)

    assert representative_gen_id == {
        0: "gen_0",
        1: "gen_1",
        2: "gen_0",  # member of group represented by index 0
        3: "gen_3",  # fallback: represents itself
    }
    assert duplicate_gen_ids[0] == ["gen_2"]
    assert duplicate_gen_ids[2] == ["gen_0"]
    assert duplicate_gen_ids[1] == []
    assert duplicate_gen_ids[3] == []  # fallback: no duplicates


def test_all_indices_covered_when_no_groups():
    results = [{"index": i, "id": f"gen_{i}"} for i in range(5)]

    representative_gen_id, duplicate_gen_ids = build_duplicate_metadata(results, groups={})

    assert set(representative_gen_id) == set(range(5))
    assert set(duplicate_gen_ids) == set(range(5))
    assert all(representative_gen_id[i] == f"gen_{i}" for i in range(5))
    assert all(duplicate_gen_ids[i] == [] for i in range(5))


def test_larger_group_lists_all_other_members_as_duplicates():
    results = [{"index": i, "id": f"gen_{i}"} for i in range(3)]
    groups = {0: [0, 1, 2]}  # all three structures are duplicates of each other

    representative_gen_id, duplicate_gen_ids = build_duplicate_metadata(results, groups)

    assert representative_gen_id == {0: "gen_0", 1: "gen_0", 2: "gen_0"}
    assert duplicate_gen_ids[0] == ["gen_1", "gen_2"]
    assert duplicate_gen_ids[1] == ["gen_0", "gen_2"]
    assert duplicate_gen_ids[2] == ["gen_0", "gen_1"]
