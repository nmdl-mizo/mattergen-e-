"""Convolutional autoencoder that compresses a per-atom DOS into a latent vector.

The released checkpoints are plain state dicts, so this definition has to match
the trained architecture exactly: the layer order fixes the parameter names that
``load_state_dict`` looks up.
"""

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_LEN = 512

ENCODER_LAYERS = [
    {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 2},
    {"in_channels": 2, "out_channels": 4, "kernel_size": 3, "stride": 2},
    {"in_channels": 4, "out_channels": 8, "kernel_size": 5, "stride": 1},
    {"in_channels": 8, "out_channels": 16, "kernel_size": 5, "stride": 1},
    {"in_channels": 16, "out_channels": 32, "kernel_size": 9, "stride": 1},
    {"in_channels": 32, "out_channels": 64, "kernel_size": 9, "stride": 1},
    {"in_channels": 64, "out_channels": 128, "kernel_size": 11, "stride": 1},
    {"in_channels": 128, "out_channels": 256, "kernel_size": 11, "stride": 1},
    {"in_channels": 256, "out_channels": 512, "kernel_size": 13, "stride": 1},
]


def make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Log1pTransform(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)


class Log1pInverse(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.expm1(y)


class ConvEncoder(nn.Module):
    """(B, 1, input_len) -> (B, latent_dim)."""

    def __init__(
        self,
        latent_dim: int,
        input_len: int = INPUT_LEN,
        bottleneck_channels: int = 64,
        pool_len: int = 32,
    ):
        super().__init__()

        layers: list[nn.Module] = [Log1pTransform()]
        for config in ENCODER_LAYERS:
            kernel_size = config["kernel_size"]
            layers.append(
                nn.Conv1d(
                    config["in_channels"],
                    config["out_channels"],
                    kernel_size=kernel_size,
                    stride=config["stride"],
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                )
            )
            layers.append(make_group_norm(config["out_channels"]))
            layers.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            out = self.conv(torch.zeros(1, 1, input_len))
        self.c, self.l = out.shape[1:]
        if pool_len > self.l:
            raise ValueError(f"pool_len ({pool_len}) must be <= conv output length ({self.l})")

        self.pre_fc = nn.Sequential(
            nn.Conv1d(self.c, bottleneck_channels, kernel_size=1, bias=False),
            make_group_norm(bottleneck_channels),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(pool_len),
        )
        self.flat_dim = bottleneck_channels * pool_len

        self.fc = nn.Linear(self.flat_dim, latent_dim)
        self.norm_z = nn.LayerNorm(latent_dim)
        self.z_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.z_scale_act = nn.Softplus(beta=1.0)
        self.act_z = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pre_fc(h).view(x.size(0), -1)
        h = self.norm_z(self.fc(h))
        return self.act_z(self.z_scale_act(self.z_scale_raw) * h)


class ConvDecoder(nn.Module):
    """(B, latent_dim) -> (B, 1, input_len)."""

    def __init__(self, latent_dim: int, c: int, length: int):
        super().__init__()
        self.c, self.length = c, length
        self.fc = nn.Linear(latent_dim, c * length)

        layers: list[nn.Module] = []
        reversed_layers = list(reversed(ENCODER_LAYERS))
        for index, config in enumerate(reversed_layers):
            kernel_size = config["kernel_size"]
            padding = (kernel_size - 1) // 2
            in_channels = config["out_channels"]
            out_channels = config["in_channels"]

            if config["stride"] > 1:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1,
                        bias=False,
                    )
                )
            else:
                layers.append(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False,
                    )
                )
            if index != len(reversed_layers) - 1:
                layers.append(nn.LeakyReLU())
        layers.extend([nn.Softplus(beta=1.0), Log1pInverse()])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.c, self.length)
        return self.deconv(h)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, input_len: int = INPUT_LEN):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, input_len)
        self.decoder = ConvDecoder(latent_dim, self.encoder.c, self.encoder.l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
