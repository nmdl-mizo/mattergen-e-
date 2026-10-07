"""End-to-end orchestration of the ADOS decoding pipeline."""

from __future__ import annotations

import argparse
import sys

from .cif_latents import check_unique_structure_ids, find_cif_files, read_latents
from .cli import parse_args
from .decoding import decode, energy_grid, load_autoencoder, resolve_device
from .diagnostics import UNPHYSICAL_PEAK_STATES_PER_EV, unphysical_mask
from .output import build_arrays, save_npz


def run(args: argparse.Namespace | None = None) -> int:
    """Decode every generated CIF under ``--structures_dir`` and return an exit code."""
    if args is None:
        args = parse_args()

    device = resolve_device(args.device)
    model, latent_dim = load_autoencoder(args.autoencoder_path, device)

    cif_files = find_cif_files(args.structures_dir)
    if not cif_files:
        raise RuntimeError(f"No CIF files found under: {args.structures_dir}")
    check_unique_structure_ids(cif_files)

    latents, records = read_latents(cif_files, args.property_prefix, latent_dim)
    dos = decode(model, latents, args.batch_size, device)
    energies = energy_grid(args.energy_min, args.energy_max, dos.shape[1])

    save_npz(args.output_npz, build_arrays(dos, energies, records, args.property_prefix))

    print(f"Decoded {len(records)} atoms from {len(cif_files)} structures on {device}.")
    print(f"Latent dimension: {latent_dim}, DOS bins: {dos.shape[1]}")
    print(f"Energy axis     : {args.energy_min} to {args.energy_max} eV (relative to E_F)")
    print(f"Archive written to: {args.output_npz}")

    n_unphysical = int(unphysical_mask(dos).sum())
    if n_unphysical:
        print(
            f"Warning: {n_unphysical} of {len(records)} atoms decoded to a DOS that is "
            f"not finite or peaks above {UNPHYSICAL_PEAK_STATES_PER_EV:g} states/eV. "
            "They are kept in the archive unchanged; filter them out before averaging "
            "or integrating.",
            file=sys.stderr,
        )
    return 0
