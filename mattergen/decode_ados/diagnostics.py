"""Report decoded spectra that cannot be physical.

The decoder ends in ``Softplus`` followed by ``expm1``, so it is exponential in
its pre-activation: a latent vector slightly outside the distribution the
autoencoder was trained on can produce a DOS many orders of magnitude above
anything physical. Such atoms are reported rather than clipped, because clipping
would silently invent a spectrum, and they are left in the archive so that
downstream code can filter on the same criterion.
"""

from __future__ import annotations

import numpy as np

UNPHYSICAL_PEAK_STATES_PER_EV = 1.0e3


def unphysical_mask(
    dos: np.ndarray, peak_threshold: float = UNPHYSICAL_PEAK_STATES_PER_EV
) -> np.ndarray:
    """Flag atoms whose decoded DOS is not finite or peaks above ``peak_threshold``."""
    finite = np.isfinite(dos)
    peaks = np.max(np.where(finite, dos, 0.0), axis=1)
    return ~finite.all(axis=1) | (peaks > peak_threshold)
