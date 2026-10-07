"""Unit tests for flagging decoded spectra that cannot be physical."""

from __future__ import annotations

import numpy as np

from mattergen.decode_ados.diagnostics import UNPHYSICAL_PEAK_STATES_PER_EV, unphysical_mask


def test_physical_spectra_are_not_flagged():
    dos = np.array([[0.0, 1.0, 5.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    assert not unphysical_mask(dos).any()


def test_large_peak_is_flagged():
    dos = np.array([[0.0, 10.0 * UNPHYSICAL_PEAK_STATES_PER_EV, 0.0]], dtype=np.float32)
    assert unphysical_mask(dos).tolist() == [True]


def test_non_finite_values_are_flagged():
    dos = np.array([[np.inf, 0.0], [np.nan, 0.0], [1.0, 2.0]], dtype=np.float32)
    assert unphysical_mask(dos).tolist() == [True, True, False]


def test_threshold_is_configurable():
    dos = np.array([[0.0, 4.0]], dtype=np.float32)
    assert unphysical_mask(dos, peak_threshold=3.0).tolist() == [True]
    assert unphysical_mask(dos, peak_threshold=5.0).tolist() == [False]
