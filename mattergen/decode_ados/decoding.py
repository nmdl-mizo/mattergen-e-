"""Load an autoencoder checkpoint and decode latent vectors in batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .autoencoder import ConvAutoencoder

LATENT_DIM_KEY = "decoder.fc.weight"


class CheckpointError(ValueError):
    """Raised when the checkpoint is not a usable autoencoder state dict."""


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_autoencoder(path: Path, device: torch.device) -> tuple[ConvAutoencoder, int]:
    """Return the autoencoder and the latent dimension read from the checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")

    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict) or LATENT_DIM_KEY not in state_dict:
        raise CheckpointError(
            f"{path} does not look like an autoencoder state dict (no {LATENT_DIM_KEY})."
        )

    latent_dim = int(state_dict[LATENT_DIM_KEY].shape[1])
    model = ConvAutoencoder(latent_dim)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), latent_dim


@torch.no_grad()
def decode(
    model: ConvAutoencoder,
    latents: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Decode ``(n_atoms, latent_dim)`` latent vectors into ``(n_atoms, n_bins)`` DOS."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    tensor = torch.as_tensor(latents, dtype=torch.float32)
    chunks = [
        model.decoder(tensor[start : start + batch_size].to(device)).squeeze(1).cpu()
        for start in range(0, len(tensor), batch_size)
    ]
    return torch.cat(chunks).numpy().astype(np.float32)


def energy_grid(energy_min: float, energy_max: float, n_bins: int) -> np.ndarray:
    if not energy_min < energy_max:
        raise ValueError(f"energy_min ({energy_min}) must be below energy_max ({energy_max})")
    return np.linspace(energy_min, energy_max, num=n_bins, dtype=np.float64)
