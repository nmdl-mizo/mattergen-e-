"""Command-line argument parsing for the ADOS decoding entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_ENERGY_MIN_EV = -15.0
DEFAULT_ENERGY_MAX_EV = 5.0
DEFAULT_PROPERTY_PREFIX = "atomic_dos"
DEFAULT_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode the per-atom density of states from the latent vectors stored "
            "as site properties in generated CIF files, using a trained "
            "convolutional autoencoder."
        )
    )
    parser.add_argument(
        "--structures_dir",
        type=Path,
        required=True,
        help="Directory containing generated CIF files (searched recursively).",
    )
    parser.add_argument(
        "--autoencoder_path",
        type=Path,
        required=True,
        help=(
            "Path to the autoencoder checkpoint (a plain state dict, e.g. "
            "mp_20_curated_-15eV_to_5eV_30dim.pt). It must be the autoencoder "
            "whose latent space the generating model was trained on."
        ),
    )
    parser.add_argument(
        "--output_npz",
        type=Path,
        default=Path("decoded_ados.npz"),
        help="Output .npz archive holding the decoded DOS and the site metadata.",
    )
    parser.add_argument(
        "--energy_min",
        type=float,
        default=DEFAULT_ENERGY_MIN_EV,
        help=(
            "Lower bound of the energy axis in eV, relative to the Fermi level. "
            "Must match the range the autoencoder was trained on; it cannot be "
            "recovered from the checkpoint."
        ),
    )
    parser.add_argument(
        "--energy_max",
        type=float,
        default=DEFAULT_ENERGY_MAX_EV,
        help="Upper bound of the energy axis in eV, relative to the Fermi level.",
    )
    parser.add_argument(
        "--property_prefix",
        type=str,
        default=DEFAULT_PROPERTY_PREFIX,
        help=(
            "Prefix of the CIF site properties holding the latent vector "
            "(<prefix>_0 ... <prefix>_<latent_dim-1>). Also used as the key of "
            "the decoded DOS in the output archive."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of atoms decoded per forward pass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to decode on. Defaults to cuda when available, else cpu.",
    )
    return parser.parse_args()
