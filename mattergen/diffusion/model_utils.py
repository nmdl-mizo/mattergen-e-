# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Any, TypeVar

import torch

from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTarget

T = TypeVar("T", bound=BatchedData)


def convert_model_out_to_score(
    *,
    model_target: ModelTarget,
    sde: SDE,
    model_out: torch.Tensor,
    batch_idx: torch.LongTensor,
    t: torch.Tensor,
    batch: Any,
) -> torch.Tensor:
    """
    Convert a model output to a score, according to the specified model_target.

    model_target: says what the model predicts.
        For example, in RFDiffusion the model predicts clean coordinates;
        in EDM the model predicts the raw noise.
    sde: corruption process
    model_out: model output
    batch_idx: indicates which sample each row of model_out belongs to
    noisy_x: noisy data
    t: diffusion timestep
    batch: noisy batch, ignored except by strange SDEs
    """
    _, std = sde.marginal_prob(
        x=torch.ones_like(model_out),
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )
    # Note the slack tolerances in test_model_utils.py: the choice of ModelTarget does make a difference.
    if model_target == ModelTarget.score_times_std:
        return model_out / std
    elif model_target == ModelTarget.logits:
        # Not really a score, but logits will be handled downstream.
        return model_out
    else:
        raise NotImplementedError

class NoiseLevelEncoding(torch.nn.Module):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        self.emb_size = d_model  # for compatibility with other embedding modules
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)
        self.register_buffer("nan_embedding", torch.zeros(d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = torch.zeros((t.shape[0], self.d_model), device=self.div_term.device)
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])

        nan_mask = torch.isnan(t)
        if nan_mask.any():
            x[nan_mask] = self.nan_embedding.unsqueeze(0).expand(nan_mask.sum(), -1)  # type: ignore[operator, call-overload]
        return self.dropout(x)


class CategoryValueEncoding(torch.nn.Module):
    """
    Learnable embedding for categorical properties.

    Maps input values (0, 1, 2, ...) directly to embedding indices.
    NaN values are automatically mapped to index (n_classes - 1).

    Args:
    ----
    n_classes: int
        Number of categories. Input values should be in [0, n_classes-1].
        NaN is mapped to index (n_classes - 1).
    d_model: int
        Embedding dimension.
    learn_nan_embedding: bool, default False
        If True, NaN embedding is learnable. If False, fixed to zero vector.
    dropout: float, default 0.0
        Dropout rate for embedding output.
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int,
        learn_nan_embedding: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.none_idx = n_classes - 1
        self.embedding = torch.nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=d_model,
            padding_idx=None if learn_nan_embedding else self.none_idx,
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)

        if not learn_nan_embedding:
            with torch.no_grad():
                self.embedding.weight[self.none_idx].fill_(0.0)

        self.dropout = torch.nn.Dropout(p=dropout)

    def _to_long_idx(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.dtype == torch.bool:
            return idx.long()

        if torch.is_floating_point(idx):
            nan_mask = torch.isnan(idx)
            if nan_mask.any():
                idx[nan_mask] = self.none_idx
            return idx.round().long()

        if idx.dtype in (torch.int32, torch.int64):
            return idx.long()

        assert False, f"Unsupported tensor dtype: {idx.dtype}"

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Convert category indices to embedding vectors.

        Parameters:
        ----------
        idx : Tensor[*,]
            Category indices (Bool/Long/Float). Float values are rounded.
            NaN values are mapped to index (n_classes - 1).

        Returns:
        -------
        Tensor[* , d_model]
            Embedding vectors.
        """
        idx_long = self._to_long_idx(idx)
        x = self.embedding(idx_long)
        return self.dropout(x)


class MlpEncoding(torch.nn.Module):
    """
    MLP encoder that maps scalar property values to a vector embedding.

    Input:  (N,) or (N, 1)
    Output: (N, d_model)
    """

    def __init__(
        self,
        d_model: int,
        input_dim: int = 1,
        hidden_dims: tuple[int, ...] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        self.d_model = d_model
        self.emb_size = d_model  # for compatibility with GemNet embedding modules
        self.input_dim = input_dim

        if hidden_dims is None:
            hidden_dims = (d_model,)

        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, d_model))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        elif x.dim() == 2 and x.size(-1) == self.input_dim:
            pass
        else:
            raise ValueError(f"Expected shape (N,) or (N, input_dim), got {tuple(x.shape)}")

        out = self.mlp(x.float())
        return out


class MlpDecoding(torch.nn.Module):
    """
    MLP decoder that maps property/site property embeddings to scalar score predictions.

    Input:  (N, d_model)
    Output: (N, output_dim)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dims: tuple[int, ...] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if hidden_dims is None:
            hidden_dims = (d_model,)

        layers: list[torch.nn.Module] = []
        in_dim = d_model
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected shape (N, d_model), got {tuple(x.shape)}")
        return self.mlp(x)
