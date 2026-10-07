# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZipFile

import ase.io
import hydra
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from mattergen.common.globals import (
    GENERATED_CRYSTALS_EXTXYZ_FILE_NAME,
    GENERATED_CRYSTALS_ZIP_FILE_NAME,
)
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import get_device
from mattergen.diffusion.lightning_module import DiffusionLightningModule

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_site_properties(
    site_properties: dict[str, np.ndarray[float, Any] | np.ndarray[list[float], Any]],
) -> dict[str, np.ndarray | list[float]]:
    out: dict[str, np.ndarray | list[float]] = {}
    for key, values in site_properties.items():
        for _, may_be_array_value in enumerate(values):
            if not isinstance(may_be_array_value, np.ndarray):
                out[key] = values
                break
            for j, value_element in enumerate(may_be_array_value):
                new_key = f"{key}_{j}"
                new_values = [*out[new_key], value_element] if new_key in out else [value_element]
                out[new_key] = new_values
    return out


def make_structure(
    lengths: np.ndarray,
    angles: np.ndarray,
    atom_types: np.ndarray,
    frac_coords: np.ndarray,
    site_properties: dict[str, np.ndarray],
) -> Structure:
    return Structure(
        lattice=Lattice.from_parameters(
            **{a: v for a, v in zip(["a", "b", "c"], lengths)},
            **{a: v for a, v in zip(["alpha", "beta", "gamma"], angles)},
        ),
        species=atom_types,  # type: ignore[arg-type]  # pymatgen accepts an ndarray of atomic numbers at runtime
        coords=frac_coords,
        coords_are_cartesian=False,
        site_properties=process_site_properties(site_properties),
    )


def load_model_diffusion(
    args: MatterGenCheckpointInfo,
) -> DiffusionLightningModule:
    assert args.load_epoch is not None
    ckpt = args.checkpoint_path
    logger.info(f"Loading model from checkpoint: {ckpt}")
    cfg = args.config
    try:
        model, incompatible_keys = DiffusionLightningModule.load_from_checkpoint_and_config(
            ckpt,
            map_location=get_device(),
            config=cfg.lightning_module,
            strict=args.strict_checkpoint_loading,
        )
    except hydra.errors.HydraException as e:
        raise
    if len(incompatible_keys.unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}.")
    if len(incompatible_keys.missing_keys) > 0:
        raise ValueError(f"Missing keys in checkpoint: {incompatible_keys.missing_keys}.")

    return model


def get_crystals_list(
    frac_coords: torch.Tensor,
    atom_types: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    num_atoms: torch.Tensor,
    bader_charge: torch.Tensor | None,
    atomic_dos: torch.Tensor | None,
    site_properties: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    # Each entry maps field name -> np.ndarray, except "site_properties" (a nested
    # dict[str, np.ndarray]) and the optional per-atom fields, which may be None.
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
        bader_charge: (num_atoms)
        site_properties: {key: (num_atoms)}
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list: list[dict[str, Any]] = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]
        cur_bader_charge = (
            bader_charge.narrow(0, start_idx, num_atom) if bader_charge is not None else None
        )
        cur_atomic_dos = (
            atomic_dos.narrow(0, start_idx, num_atom) if atomic_dos is not None else None
        )
        cur_site_properties = {
            k: v.narrow(0, start_idx, num_atom).detach().cpu().numpy()
            for k, v in site_properties.items()
        }
        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
                "bader_charge": (
                    cur_bader_charge.detach().cpu().numpy()
                    if cur_bader_charge is not None
                    else None
                ),
                "atomic_dos": (
                    cur_atomic_dos.detach().cpu().numpy() if cur_atomic_dos is not None else None
                ),
                "site_properties": cur_site_properties,
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


def save_structures(output_path: Path, structures: Sequence[Structure]) -> None:
    """Save structures to disk in a extxyz file and a compressed zip file containing cif files.

    Args:
        output_path: path to a directory where the results are written.
        structures: sequence of structures.
    """
    ase_atoms = []
    for struct in structures:
        atoms = AseAtomsAdaptor.get_atoms(struct)
        for prop, values in struct.site_properties.items():
            atoms.set_array(prop, np.asarray(values))
        ase_atoms.append(atoms)

    try:
        ase.io.write(output_path / GENERATED_CRYSTALS_EXTXYZ_FILE_NAME, ase_atoms)

        with ZipFile(output_path / GENERATED_CRYSTALS_ZIP_FILE_NAME, "w") as zip_obj:
            for ix, (struct, atoms) in enumerate(zip(structures, ase_atoms)):
                # CifWriter (not ase.io.write) so that site properties end up in the CIF.
                CifWriter(struct, write_site_properties=True).write_file(f"/tmp/gen_{ix}.cif")
                zip_obj.write(f"/tmp/gen_{ix}.cif", arcname=f"gen_{ix}.cif")
    except IOError as e:
        print(f"Got error {e} writing the generated structures to disk.")
