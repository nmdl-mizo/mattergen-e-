"""Unit tests for CSV writing and CIF copying (mattergen.evaluate_un.output)."""

import csv

from mattergen.evaluate_un.labeling import NOVEL_LABEL, UNIQUE_LABEL, compute_structure_labels
from mattergen.evaluate_un.output import CSV_FIELDNAMES, copy_cifs_by_mask, save_csv


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sample_results():
    return [
        # in reference, representative of group {0, 1}            -> "U"
        {
            "index": 0,
            "id": "gen_0",
            "cif_path": "/x/gen_0.cif",
            "in_reference": True,
            "matched_material_ids": ["mp-1", "mp-2"],
        },
        # in reference, non-representative duplicate of group {0, 1} -> ""
        {
            "index": 1,
            "id": "gen_1",
            "cif_path": "/x/gen_1.cif",
            "in_reference": True,
            "matched_material_ids": [],
        },
        # novel, not duplicated                                   -> "UN"
        {
            "index": 2,
            "id": "gen_2",
            "cif_path": "/x/gen_2.cif",
            "in_reference": False,
            "matched_material_ids": [],
        },
        # novel, non-representative duplicate                     -> "N"
        {
            "index": 3,
            "id": "gen_3",
            "cif_path": "/x/gen_3.cif",
            "in_reference": False,
            "matched_material_ids": [],
        },
    ]


def test_save_csv_header_matches_fieldnames_and_ends_with_structure(tmp_path):
    out = tmp_path / "out.csv"
    save_csv(results=[], path=out, representative_gen_id={}, duplicate_gen_ids={})

    with out.open(newline="", encoding="utf-8") as f:
        header = f.readline().strip()

    assert header == ",".join(CSV_FIELDNAMES)
    assert CSV_FIELDNAMES[-1] == "structure"


def test_save_csv_writes_all_four_structure_labels(tmp_path):
    results = _sample_results()
    representative_gen_id = {0: "gen_0", 1: "gen_0", 2: "gen_2", 3: "gen_4"}
    duplicate_gen_ids = {0: ["gen_1"], 1: ["gen_0"], 2: [], 3: ["gen_4"]}

    out = tmp_path / "out.csv"
    save_csv(
        results=results,
        path=out,
        representative_gen_id=representative_gen_id,
        duplicate_gen_ids=duplicate_gen_ids,
    )
    rows = _read_rows(out)

    # The identifier column is written as "id" (renamed from "gen_id").
    assert "id" in rows[0] and "gen_id" not in rows[0]
    assert [r["id"] for r in rows] == ["gen_0", "gen_1", "gen_2", "gen_3"]

    # The new structure column covers all four expected values, in row order.
    assert [r["structure"] for r in rows] == ["U", "", "UN", "N"]
    assert {r["structure"] for r in rows} == {"", "U", "N", "UN"}

    # Derived bookkeeping columns are computed as before.
    assert rows[0]["matched_material_ids"] == "mp-1;mp-2"
    assert rows[0]["is_duplicated"] == "True"
    assert rows[2]["is_duplicated"] == "False"
    assert rows[0]["representative_gen_id"] == "gen_0"
    assert rows[3]["representative_gen_id"] == "gen_4"
    assert rows[3]["duplicate_gen_ids"] == "gen_4"
    assert rows[2]["duplicate_gen_ids"] == ""


def test_save_csv_creates_missing_parent_directory(tmp_path):
    out = tmp_path / "nested" / "deeper" / "out.csv"
    save_csv(
        results=_sample_results()[:1],
        path=out,
        representative_gen_id={0: "gen_0"},
        duplicate_gen_ids={0: []},
    )
    assert out.is_file()


def _make_cif_files(directory, gen_ids):
    directory.mkdir(parents=True, exist_ok=True)
    cif_files = []
    for gen_id in gen_ids:
        p = directory / f"{gen_id}.cif"
        p.write_text(f"data_{gen_id}\n")
        cif_files.append(p)
    return cif_files


def test_copy_cifs_by_mask_copies_only_masked(tmp_path):
    cif_files = _make_cif_files(tmp_path / "src", ["gen_0", "gen_1", "gen_2", "gen_3"])
    mask = [True, False, True, False]
    out = tmp_path / "out"

    count = copy_cifs_by_mask(cif_files, mask, out)

    assert count == 2
    assert sorted(p.name for p in out.glob("*.cif")) == ["gen_0.cif", "gen_2.cif"]
    assert (out / "gen_0.cif").read_text() == "data_gen_0\n"


def test_copy_cifs_by_mask_handles_none_matching(tmp_path):
    cif_files = _make_cif_files(tmp_path / "src", ["gen_0"])

    count = copy_cifs_by_mask(cif_files, [False], output_dir=tmp_path / "out")

    assert count == 0
    assert list((tmp_path / "out").glob("*.cif")) == []


def test_copy_cifs_by_mask_removes_preexisting_cifs(tmp_path):
    """Stale CIFs from a previous run are deleted before the new ones are copied.

    The output directory must end up containing exactly the CIFs selected for the
    current run; only ``*.cif`` files are removed, other files are preserved.
    """
    cif_files = _make_cif_files(tmp_path / "src", ["gen_0", "gen_1", "gen_2"])

    out = tmp_path / "out"
    out.mkdir()
    # Simulate leftovers from a previous run plus an unrelated non-CIF file.
    (out / "stale_gen_99.cif").write_text("old structure")
    (out / "keep.txt").write_text("not a cif")

    count = copy_cifs_by_mask(cif_files, [True, False, True], out)

    assert count == 2
    # Only the freshly selected CIFs remain; the stale CIF is gone.
    assert sorted(p.name for p in out.glob("*.cif")) == ["gen_0.cif", "gen_2.cif"]
    assert not (out / "stale_gen_99.cif").exists()
    # Non-CIF files are left untouched.
    assert (out / "keep.txt").read_text() == "not a cif"


def test_copy_cifs_by_mask_empty_selection_clears_existing_cifs(tmp_path):
    """When the current run selects nothing, the directory is emptied of CIFs."""
    cif_files = _make_cif_files(tmp_path / "src", ["gen_0"])

    out = tmp_path / "out"
    out.mkdir()
    (out / "stale_gen_99.cif").write_text("old structure")

    count = copy_cifs_by_mask(cif_files, [False], out)

    assert count == 0
    assert list(out.glob("*.cif")) == []


def test_u_and_n_directories_receive_correct_cifs(tmp_path):
    """End-to-end (pure-logic) check of the U / N directory selection.

    The U directory must get every "U" or "UN" structure, the N directory every
    "N" or "UN" structure, so a "UN" structure lands in *both* directories.
    """
    cif_files = _make_cif_files(tmp_path / "src", ["gen_0", "gen_1", "gen_2", "gen_3"])
    results = [
        {"index": 0, "id": "gen_0", "in_reference": True},  # "U"
        {"index": 1, "id": "gen_1", "in_reference": True},  # ""
        {"index": 2, "id": "gen_2", "in_reference": False},  # "UN"
        {"index": 3, "id": "gen_3", "in_reference": False},  # "N"
    ]
    representative_gen_id = {0: "gen_0", 1: "gen_0", 2: "gen_2", 3: "gen_4"}
    duplicate_gen_ids = {0: ["gen_1"], 1: ["gen_0"], 2: [], 3: ["gen_4"]}

    labels = compute_structure_labels(results, representative_gen_id, duplicate_gen_ids)
    assert labels == ["U", "", "UN", "N"]

    u_dir = tmp_path / "u_out"
    n_dir = tmp_path / "n_out"
    copied_u = copy_cifs_by_mask(
        cif_files, [UNIQUE_LABEL in label for label in labels], u_dir
    )
    copied_n = copy_cifs_by_mask(
        cif_files, [NOVEL_LABEL in label for label in labels], n_dir
    )

    assert copied_u == 2
    assert copied_n == 2
    # U directory: "U" (gen_0) and "UN" (gen_2)
    assert sorted(p.name for p in u_dir.glob("*.cif")) == ["gen_0.cif", "gen_2.cif"]
    # N directory: "UN" (gen_2) and "N" (gen_3)
    assert sorted(p.name for p in n_dir.glob("*.cif")) == ["gen_2.cif", "gen_3.cif"]
    # The "UN" structure appears in BOTH directories.
    assert (u_dir / "gen_2.cif").is_file()
    assert (n_dir / "gen_2.cif").is_file()


def test_un_directory_receives_only_unique_and_novel(tmp_path):
    """The UN directory must get only structures that are both unique and novel.

    A structure is copied to the UN directory exactly when its label is "UN"
    (i.e. contains both "U" and "N"); plain "U", plain "N" and "" are excluded.
    """
    cif_files = _make_cif_files(tmp_path / "src", ["gen_0", "gen_1", "gen_2", "gen_3"])
    results = [
        {"index": 0, "id": "gen_0", "in_reference": True},  # "U"
        {"index": 1, "id": "gen_1", "in_reference": True},  # ""
        {"index": 2, "id": "gen_2", "in_reference": False},  # "UN"
        {"index": 3, "id": "gen_3", "in_reference": False},  # "N"
    ]
    representative_gen_id = {0: "gen_0", 1: "gen_0", 2: "gen_2", 3: "gen_4"}
    duplicate_gen_ids = {0: ["gen_1"], 1: ["gen_0"], 2: [], 3: ["gen_4"]}

    labels = compute_structure_labels(results, representative_gen_id, duplicate_gen_ids)
    assert labels == ["U", "", "UN", "N"]

    un_dir = tmp_path / "un_out"
    copied_un = copy_cifs_by_mask(
        cif_files,
        [UNIQUE_LABEL in label and NOVEL_LABEL in label for label in labels],
        un_dir,
    )

    assert copied_un == 1
    # Only the "UN" structure (gen_2) lands in the UN directory.
    assert sorted(p.name for p in un_dir.glob("*.cif")) == ["gen_2.cif"]
