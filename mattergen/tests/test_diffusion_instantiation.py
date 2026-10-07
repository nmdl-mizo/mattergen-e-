# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end smoke test: every top-level training config trains for one batch.

This is the test that catches a config drifting away from the code (a renamed module,
a removed field, a mismatched embedding size): it instantiates the data module, the
score model and the trainer exactly as ``mattergen-train`` would, and runs a single
optimisation step.

Each config is skipped rather than failed when its preprocessed dataset is not present,
so the suite stays meaningful on a machine that only has some of the datasets.
"""

import os
from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.scripts.run import mattergen_main

CONFIG_DIR = Path(MODELS_PROJECT_ROOT) / "conf"

# Every top-level config, i.e. every value that can be passed to `mattergen-train
# --config-name=...`. Nested configs (trainer/, data_module/, ...) are pulled in through
# the `defaults` list of these files and are covered transitively.
CONFIG_NAMES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

# `finetune` starts from a pre-trained checkpoint (adapter.pretrained_name), which is not
# available offline, so it cannot be smoke-tested this way.
CONFIG_NAMES_REQUIRING_CHECKPOINT = {"finetune"}

TRAINABLE_CONFIG_NAMES = [c for c in CONFIG_NAMES if c not in CONFIG_NAMES_REQUIRING_CHECKPOINT]


def _overrides(default_root_dir: Path) -> list[str]:
    """Config overrides that make a training run short, CPU-only and self-contained."""
    return [
        # Keep the run to a single batch and a single (tiny) model block.
        "trainer.max_epochs=1",
        "+trainer.overfit_batches=1",
        "trainer.check_val_every_n_epoch=1",
        "+trainer.limit_val_batches=1",
        "+trainer.limit_test_batches=1",
        "lightning_module.diffusion_module.model.gemnet.num_blocks=1",
        "lightning_module.diffusion_module.model.gemnet.max_neighbors=5",
        "lightning_module.diffusion_module.model.hidden_dim=16",
        "data_module.batch_size.train=8",
        "data_module.batch_size.val=8",
        "data_module.batch_size.test=8",
        "trainer.accelerator=cpu",
        # wandb does not work in CI.
        "~trainer.logger",
        # Without these two the run would pick up (and try to load) checkpoints from an
        # earlier real training run, and would write its own checkpoints into the
        # repository. Both make the test depend on -- and pollute -- its environment.
        "auto_resume=False",
        f"+trainer.default_root_dir={default_root_dir}",
    ]


def _compose(config_name: str, default_root_dir: Path) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return hydra.compose(config_name=config_name, overrides=_overrides(default_root_dir))


def test_all_top_level_configs_are_covered():
    """Guards against a new top-level config being added without a smoke test."""
    assert set(TRAINABLE_CONFIG_NAMES) | CONFIG_NAMES_REQUIRING_CHECKPOINT == set(CONFIG_NAMES)
    assert TRAINABLE_CONFIG_NAMES, f"no trainable configs found in {CONFIG_DIR}"


@pytest.mark.slow
@pytest.mark.parametrize("config_name", TRAINABLE_CONFIG_NAMES)
def test_train_on_one_batch(config_name: str, tmp_path: Path) -> None:
    """The model can be instantiated from `config_name` and trained on one batch."""
    config = _compose(config_name, default_root_dir=tmp_path)

    dataset_dir = Path(config.data_module.root_dir)
    if not dataset_dir.is_dir():
        pytest.skip(
            f"preprocessed dataset for '{config_name}' not found at {dataset_dir}; "
            "run csv-to-dataset first"
        )

    # `mattergen_main` returns nothing; reaching the next line means data loading, model
    # instantiation, the forward/backward pass and validation all ran without raising.
    mattergen_main(config)

    # Everything the run produced must land in the sandbox, not in the repository. This
    # also confirms the `default_root_dir` override took effect.
    assert any(tmp_path.iterdir()), "training run produced no output under default_root_dir"


@pytest.mark.parametrize("config_name", TRAINABLE_CONFIG_NAMES)
def test_config_composes_and_instantiates_score_model(config_name: str, tmp_path: Path) -> None:
    """Fast counterpart of the training test: the config resolves and the model builds.

    This needs no dataset, so it runs everywhere and catches the majority of config
    regressions (bad interpolation, renamed target, wrong argument name).
    """
    config = _compose(config_name, default_root_dir=tmp_path)

    model = hydra.utils.instantiate(config.lightning_module.diffusion_module.model)

    assert isinstance(model, __import__("torch").nn.Module)
    assert os.fspath(config.data_module.root_dir)
