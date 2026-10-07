# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional

from mattergen.diffusion.losses import SummedFieldLoss, denoising_score_matching
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import FieldLoss, d3pm_loss
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss


class MaterialsLoss(SummedFieldLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        include_charge: bool = False,
        include_atomic_dos: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ):
        model_targets = {"pos": ModelTarget.score_times_std, "cell": ModelTarget.score_times_std}
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss,
                reduce=reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")
            loss_fns["atomic_numbers"] = partial(
                d3pm_loss,
                reduce=reduce,
                d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            )
        if include_charge:
            model_targets["bader_charge"] = ModelTarget.score_times_std
            self.fields_to_score.append("bader_charge")
            loss_fns["bader_charge"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["bader_charge"],
            )
        if include_atomic_dos:
            model_targets["atomic_dos"] = ModelTarget.score_times_std
            self.fields_to_score.append("atomic_dos")
            loss_fns["atomic_dos"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["atomic_dos"],
            )
        self.reduce = reduce
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            model_targets=model_targets,
        )
