# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch_scatter import scatter_add

from mattergen.common.diffusion import corruption as sde_lib
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition
from mattergen.diffusion.corruption.corruption import Corruption, maybe_expand
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling import predictors_correctors as pc
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor

SampleAndMean = tuple[torch.Tensor, torch.Tensor]

class LatticeAncestralSamplingPredictor(AncestralSamplingPredictor):
    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.LatticeVPSDE)

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x,
            t=t,
            dt=dt,
            batch_idx=batch_idx,
            batch=batch,
        )
        # mean = (x + score * beta**2 - limit_mean)/(1-beta) + limit_mean
        # <=> mean = x / (1-beta) + score * beta**2 / (1-beta) + limit_mean * (1 - 1/(1-beta))
        # => mean_coeff = 1 - x_coeff = 1 - 1/(1-beta)
        mean_coeff = 1 - x_coeff
        # Sample random noise.
        z = sde_lib.make_noise_symmetric_preserve_variance(torch.randn_like(x_coeff))
        assert hasattr(self.corruption, "get_limit_mean")  # mypy
        mean = (
            x_coeff * x
            + score_coeff * score
            + mean_coeff * self.corruption.get_limit_mean(x=x, batch=batch)
        )
        sample = mean + std * z
        return sample, mean


# create a langevin corrector that accepts LatticeVPSDE
class LatticeLangevinDiffCorrector(pc.LangevinCorrector):
    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.LatticeVPSDE)

    def step_given_score(
        self,
        *,
        x: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> SampleAndMean:
        assert isinstance(self.corruption, sde_lib.LatticeVPSDE)
        alpha = self.get_alpha(t, dt=dt)
        snr = self.snr
        noise = torch.randn_like(x)
        noise = sde_lib.make_noise_symmetric_preserve_variance(noise)

        # [batch_size, ] or [num_atoms, ] if batch_idx is not None
        grad_norm_square = torch.square(score).reshape(score.shape[0], -1).sum(dim=1)
        noise_norm_square = torch.square(noise).reshape(noise.shape[0], -1).sum(dim=1)
        # Average over items, leading to scalars.
        grad_norm = grad_norm_square.sqrt().mean()
        noise_norm = noise_norm_square.sqrt().mean()

        # If gradient is zero (i.e., we are sampling from an improper distribution that's flat over the whole of R^n)
        # the step_size blows up. Clip step_size to avoid this.
        # The EGNN reports zero scores when there are no edges between nodes.
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, self.max_step_size)
        step_size[grad_norm == 0, :] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + torch.sqrt(step_size * 2) * noise

        x = compute_lattice_polar_decomposition(x)
        mean = compute_lattice_polar_decomposition(mean)
        return x, mean


def apply_charge_neutrality(
    x: torch.Tensor, batch_idx: torch.LongTensor | None, t: torch.Tensor
) -> torch.Tensor:
    """Apply time-weighted charge neutrality per structure (for use from corrector and predictor).

    Strong at t~0, weak at t~1. Used so that predictor output is also neutralized,
    avoiding drift from neutrality on the final step.
    """
    weight = (1.0 - t).clamp(0.0, 1.0)

    if batch_idx is None:
        weight_scalar = weight.squeeze().mean() if weight.numel() > 1 else weight.item()
        total_charge = x.sum()
        delta = total_charge / x.shape[0]
        return x - weight_scalar * delta

    weight_expanded = maybe_expand(weight, batch_idx, like=x)
    total_charges = scatter_add(x.squeeze(-1) if x.dim() > 1 else x, batch_idx, dim=0)
    num_atoms_per_crystal = scatter_add(torch.ones_like(batch_idx, dtype=x.dtype), batch_idx, dim=0)
    avg_charges = total_charges / num_atoms_per_crystal

    if x.dim() > 1:
        adjustment = avg_charges[batch_idx].unsqueeze(-1)
        return x - weight_expanded.unsqueeze(-1) * adjustment
    return x - weight_expanded * avg_charges[batch_idx]


class ChargeAncestralSamplingPredictor(AncestralSamplingPredictor):
    """Ancestral sampling predictor for charge fields using StandardVPSDE."""

    def __init__(self, *args, apply_charge_neutrality_constraint: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_charge_neutrality_constraint = apply_charge_neutrality_constraint

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.StandardVPSDE)

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        if x.dim() == 2:
            x = x.squeeze(-1)
        elif score.dim() == 2:
            score = score.squeeze(-1)

        x_coeff, score_coeff, std = self._get_coeffs(
            x=x,
            t=t,
            dt=dt,
            batch_idx=batch_idx,
            batch=batch,
        )
        mean = x_coeff * x + score_coeff * score
        z = torch.randn_like(x)
        sample = mean + std * z
        if self.apply_charge_neutrality_constraint:
            sample = apply_charge_neutrality(sample, batch_idx, t)
            mean = apply_charge_neutrality(mean, batch_idx, t)
        return sample, mean


class ChargeNeutralLangevinCorrector(pc.LangevinCorrector):
    """Langevin corrector for charge fields with optional charge neutrality constraint.

    This corrector can optionally ensure that the total charge within each crystal structure
    remains close to zero (charge neutral) by applying a constraint after each step.
    """

    def __init__(self, *args, apply_charge_neutrality_constraint: bool = True, **kwargs):
        """Initialize the charge corrector with optional neutrality constraint.

        Args:
            corruption: corruption process
            score_fn: score function
            n_steps: number of Langevin steps at each noise level
            snr: signal-to-noise ratio
            max_step_size: largest coefficient that the score can be multiplied by for each Langevin step
            apply_charge_neutrality_constraint: whether to apply charge neutrality constraint
        """
        super().__init__(*args, **kwargs)
        self.apply_charge_neutrality_constraint = apply_charge_neutrality_constraint

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.StandardVPSDE)

    def step_given_score(
        self,
        *,
        x: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> SampleAndMean:
        alpha = self.get_alpha(t, dt=dt)
        snr = self.snr

        if x.dim() == 2:
            x = x.squeeze(-1)
        elif score.dim() == 2:
            score = score.squeeze(-1)
        noise = torch.randn_like(x)

        grad_norm_square = torch.square(score).reshape(score.shape[0], -1).sum(dim=1)
        noise_norm_square = torch.square(noise).reshape(noise.shape[0], -1).sum(dim=1)
        grad_norm = grad_norm_square.sqrt().mean()
        noise_norm = noise_norm_square.sqrt().mean()

        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, self.max_step_size)
        step_size[grad_norm == 0, :] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + torch.sqrt(step_size * 2) * noise

        # Apply charge neutrality constraint if enabled
        if self.apply_charge_neutrality_constraint:
            x = apply_charge_neutrality(x, batch_idx, t)
            mean = apply_charge_neutrality(mean, batch_idx, t)

        return x, mean


def apply_weighted_clamp(
    x: torch.Tensor,
    weight: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """Weighted clamp: weight * clamp(x) + (1 - weight) * x.

    Used for time-dependent clamping (e.g. atomic DOS in [-1, 1]).
    weight must be broadcastable to x; typically (batch_size,) expanded to (num_atoms,) or (num_atoms, 1).
    """
    x_clamped = torch.clamp(x, min=clamp_min, max=clamp_max)
    return weight * x_clamped + (1.0 - weight) * x


class AtomicDosAncestralSamplingPredictor(AncestralSamplingPredictor):
    """Ancestral sampling predictor for atomic DOS latent vector fields using StandardVPSDE."""

    def __init__(self, *args, clamp_min: float = -1.0, clamp_max: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.StandardVPSDE)

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        if x.dim() == 2:
            x = x.squeeze(-1)
        elif score.dim() == 2:
            score = score.squeeze(-1)

        x_coeff, score_coeff, std = self._get_coeffs(
            x=x,
            t=t,
            dt=dt,
            batch_idx=batch_idx,
            batch=batch,
        )
        mean = x_coeff * x + score_coeff * score
        z = torch.randn_like(x)
        sample = mean + std * z

        weight = (1.0 - t).clamp(0.0, 1.0)
        weight_expanded = maybe_expand(weight, batch_idx, like=sample)
        sample = apply_weighted_clamp(sample, weight_expanded, self.clamp_min, self.clamp_max)
        mean = apply_weighted_clamp(mean, weight_expanded, self.clamp_min, self.clamp_max)

        return sample, mean


class AtomicDosLangevinCorrector(pc.LangevinCorrector):
    """Langevin corrector for atomic DOS latent vector fields."""

    def __init__(self, *args, clamp_min: float = -1.0, clamp_max: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.StandardVPSDE)

    def step_given_score(
        self,
        *,
        x: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> SampleAndMean:
        alpha = self.get_alpha(t, dt=dt)
        snr = self.snr

        noise = torch.randn_like(x)

        grad_norm_square = torch.square(score).reshape(score.shape[0], -1).sum(dim=1)
        noise_norm_square = torch.square(noise).reshape(noise.shape[0], -1).sum(dim=1)
        grad_norm = grad_norm_square.sqrt().mean()
        noise_norm = noise_norm_square.sqrt().mean()

        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, self.max_step_size)
        step_size[grad_norm == 0, :] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)

        mean = x + step_size * score
        sample = mean + torch.sqrt(step_size * 2) * noise

        # Apply time-dependent weighted clamping: strong at t~0 (clean), weak at t~1 (noisy)
        weight = (1.0 - t).clamp(0.0, 1.0)
        weight_expanded = maybe_expand(weight, batch_idx, like=sample)
        sample = apply_weighted_clamp(sample, weight_expanded, self.clamp_min, self.clamp_max)
        mean = apply_weighted_clamp(mean, weight_expanded, self.clamp_min, self.clamp_max)

        return sample, mean
