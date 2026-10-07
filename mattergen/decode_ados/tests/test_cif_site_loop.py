"""Unit tests for the raw ``_atom_site_`` loop parser (no pymatgen required)."""

from __future__ import annotations

import pytest

from mattergen.decode_ados.cif_site_loop import CifLoopError, parse_atom_site_loop

WRAPPED_CIF = """# generated using pymatgen
data_AlGa
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.5
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_occupancy
 _atom_site_atomic_dos_0
 _atom_site_atomic_dos_1
  Al  Al0  1  -0.1
  0.2
  Ga  Ga1  1  0.3
  -0.4
"""


def test_columns_are_read_across_wrapped_rows():
    columns = parse_atom_site_loop(WRAPPED_CIF)
    assert columns["label"] == ["Al0", "Ga1"]
    assert columns["type_symbol"] == ["Al", "Ga"]
    assert columns["atomic_dos_0"] == ["-0.1", "0.3"]
    assert columns["atomic_dos_1"] == ["0.2", "-0.4"]


def test_preceding_loops_are_skipped():
    columns = parse_atom_site_loop(WRAPPED_CIF)
    assert "as_xyz" not in columns
    assert set(columns) == {"type_symbol", "label", "occupancy", "atomic_dos_0", "atomic_dos_1"}


def test_quoted_values_stay_single_tokens():
    cif = (
        "loop_\n"
        " _atom_site_label\n"
        " _atom_site_note\n"
        "  Al0  'a note'\n"
    )
    assert parse_atom_site_loop(cif)["note"] == ["a note"]


def test_missing_loop_is_rejected():
    with pytest.raises(CifLoopError):
        parse_atom_site_loop("data_x\n_cell_length_a 3.5\n")


def test_ragged_row_is_rejected():
    cif = "loop_\n _atom_site_label\n _atom_site_occupancy\n  Al0  1\n  Ga1\n"
    with pytest.raises(CifLoopError):
        parse_atom_site_loop(cif)
