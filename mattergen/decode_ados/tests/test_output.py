"""Unit tests for assembling and writing the output archive."""

from __future__ import annotations

import numpy as np
import pytest

from mattergen.decode_ados.output import build_arrays, save_npz
from mattergen.decode_ados.records import SiteRecord

RECORDS = [SiteRecord("gen_0", 0, "Al", "Al0"), SiteRecord("gen_0", 1, "Ga", "Ga1")]
DOS = np.arange(8, dtype=np.float32).reshape(2, 4)
ENERGIES = np.linspace(-15.0, 5.0, 4)


def test_arrays_are_row_aligned():
    arrays = build_arrays(DOS, ENERGIES, RECORDS, "atomic_dos")
    assert set(arrays) == {"atomic_dos", "energies", "structure_id", "site_index", "element", "label"}
    assert arrays["atomic_dos"].shape == (2, 4)
    assert arrays["label"].tolist() == ["Al0", "Ga1"]
    assert arrays["site_index"].tolist() == [0, 1]


def test_row_count_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_arrays(DOS, ENERGIES, RECORDS[:1], "atomic_dos")


def test_bin_count_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_arrays(DOS, ENERGIES[:3], RECORDS, "atomic_dos")


def test_reserved_key_is_rejected():
    with pytest.raises(ValueError):
        build_arrays(DOS, ENERGIES, RECORDS, "energies")


def test_archive_round_trips(tmp_path):
    path = tmp_path / "out" / "decoded.npz"
    save_npz(path, build_arrays(DOS, ENERGIES, RECORDS, "atomic_dos"))

    assert path.is_file()
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["atomic_dos"], DOS)
        np.testing.assert_allclose(archive["energies"], ENERGIES)
        assert archive["element"].tolist() == ["Al", "Ga"]
