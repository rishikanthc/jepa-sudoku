from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation, TransformerConfig

from .csv_logger import SFTMetricCSVLogger
from .sft_data import DifficultyRange, SFTDataConfig, SFTLightningDataModule
from .sft_module import FrozenEncoderSFTModule, SFTTrainConfig


@dataclass(frozen=True)
class SFTExperimentResult:
    history: list[float]
    is_global_zero: bool
    checkpoint_path: str | None = None
    test_metrics: dict[str, float] | None = None


class CurriculumAwareEarlyStopping(EarlyStopping):
    def __init__(
        self,
        *,
        curriculum_enabled: bool,
        curriculum_max_mask: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_max_mask = int(curriculum_max_mask)

    def _curriculum_at_max_difficulty(self, trainer: pl.Trainer) -> bool:
        data_module = trainer.datamodule
        if data_module is None or not hasattr(data_module, "current_num_cells_to_mask"):
            return True
        return int(data_module.current_num_cells_to_mask) >= self.curriculum_max_mask

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.curriculum_enabled and not self._curriculum_at_max_difficulty(trainer):
            return
        super().on_validation_end(trainer, pl_module)


def _build_transformer_config(config: DictConfig) -> TransformerConfig:
    return TransformerConfig(
        context_size=config.context_size,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )


def _load_pretrain_config(config_name: str) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        config = compose(config_name=config_name)
    else:
        config_dir = str((Path(__file__).resolve().parents[2] / "configs"))
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name=config_name)
    if "model" not in config or "ssp" not in config:
        raise ValueError(
            f"Pretrain config '{config_name}' must define both 'model' and 'ssp' sections."
        )
    return config


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint state_dict must be a dict.")
    return state_dict


def build_sft_data_config(config: DictConfig) -> SFTDataConfig:
    return SFTDataConfig(
        dataset_path=config.data.dataset_path,
        num_samples=config.data.num_samples,
        seed=config.data.seed,
        batch_size=config.data.batch_size,
        eval_batch_size=config.data.eval_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
        shuffle=config.data.shuffle,
        randomize_train_mask_per_access=config.data.randomize_train_mask_per_access,
        val_test_randomize_mask_per_access=config.data.val_test_randomize_mask_per_access,
        train_num_cells_to_mask=config.data.train_num_cells_to_mask,
        difficulty_range=DifficultyRange(
            start=config.data.difficulty_range.start,
            end=config.data.difficulty_range.end,
            step=config.data.difficulty_range.step,
        ),
    )


def build_sft_train_config(config: DictConfig) -> SFTTrainConfig:
    return SFTTrainConfig(
        learning_rate=config.training.learning_rate,
        curriculum_enabled=config.curriculum.enabled,
        curriculum_mode=config.curriculum.mode,
        curriculum_monitor=config.curriculum.monitor,
        curriculum_step=config.curriculum.step,
        curriculum_patience=config.curriculum.patience,
        curriculum_min_delta=config.curriculum.min_delta,
        curriculum_max_mask=config.curriculum.max_mask,
    )


def build_sft_data_module(config: DictConfig) -> SFTLightningDataModule:
    return SFTLightningDataModule(
        config=build_sft_data_config(config),
        curriculum_enabled=config.curriculum.enabled,
        curriculum_mode=config.curriculum.mode,
        curriculum_max_mask=config.curriculum.max_mask,
        curriculum_num_steps=config.curriculum.num_steps,
    )


def build_frozen_encoder_and_decoder(
    config: DictConfig,
) -> tuple[SudokuRepresentation, Encoder, Predictor]:
    pretrain_config = _load_pretrain_config(config.pretrained.config_name)
    encoder_transformer_config = _build_transformer_config(pretrain_config.model)
    decoder_transformer_config = _build_transformer_config(config.decoder_model)
    if encoder_transformer_config.d_model != decoder_transformer_config.d_model:
        raise ValueError(
            "Decoder d_model must match pretrained encoder d_model for cross-attention."
        )
    if pretrain_config.ssp.dim != encoder_transformer_config.d_model:
        raise ValueError(
            "Pretrain config ssp.dim must match pretrained encoder d_model."
        )

    representation = SudokuRepresentation(
        d_model=encoder_transformer_config.d_model,
        seed=pretrain_config.ssp.seed,
    )
    encoder = Encoder(encoder_transformer_config, representation=representation)
    decoder = Predictor(decoder_transformer_config, representation=representation)

    state_dict = _load_state_dict_from_checkpoint(config.pretrained.checkpoint_path)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    representation_state = {
        key.removeprefix("representation."): value
        for key, value in state_dict.items()
        if key.startswith("representation.")
    }
    if not encoder_state:
        raise ValueError("Checkpoint is missing encoder weights.")
    if not representation_state:
        raise ValueError("Checkpoint is missing representation state.")
    encoder.load_state_dict(encoder_state, strict=True)
    representation.load_state_dict(representation_state, strict=True)
    return representation, encoder, decoder


def build_sft_lightning_module(
    config: DictConfig,
    *,
    data_module: SFTLightningDataModule,
) -> FrozenEncoderSFTModule:
    representation, encoder, decoder = build_frozen_encoder_and_decoder(config)
    return FrozenEncoderSFTModule(
        encoder=encoder,
        decoder=decoder,
        representation=representation,
        config=build_sft_train_config(config),
        val_difficulties=data_module.config.difficulty_range.levels(),
        test_difficulties=data_module.config.difficulty_range.levels(),
        checkpoint_monitor=config.checkpoint.monitor,
        checkpoint_mode=config.checkpoint.mode,
    )


def _build_callbacks(config: DictConfig) -> list[pl.Callback]:
    early_stopping = CurriculumAwareEarlyStopping(
        curriculum_enabled=config.curriculum.enabled,
        curriculum_max_mask=config.curriculum.max_mask,
        monitor=config.early_stopping.monitor,
        mode=config.early_stopping.mode,
        patience=config.early_stopping.patience,
        min_delta=config.early_stopping.min_delta,
        check_on_train_epoch_end=False,
    )
    csv_callback = SFTMetricCSVLogger(config.logging.csv_path)
    return [early_stopping, csv_callback]


def build_sft_trainer(config: DictConfig) -> pl.Trainer:
    val_check_interval = config.trainer.val_check_interval
    if val_check_interval == "null":
        val_check_interval = None
    check_val_every_n_epoch = config.trainer.check_val_every_n_epoch
    if check_val_every_n_epoch == "null":
        check_val_every_n_epoch = None

    # Lightning runs validation solely by train-step interval when
    # `check_val_every_n_epoch=None` and `val_check_interval` is an integer.
    if isinstance(val_check_interval, int) and val_check_interval > 0:
        check_val_every_n_epoch = None
    elif val_check_interval is None and check_val_every_n_epoch is None:
        # Preserve Lightning's normal epoch-based validation unless the user
        # explicitly opts into step-based validation.
        val_check_interval = 1.0
        check_val_every_n_epoch = 1

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        precision=config.trainer.precision,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        max_epochs=config.training.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=False,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        fast_dev_run=config.trainer.fast_dev_run,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        callbacks=_build_callbacks(config),
        logger=False,
        enable_model_summary=False,
    )
    return trainer


def run_sft_experiment(config: DictConfig) -> SFTExperimentResult:
    torch.set_float32_matmul_precision(config.trainer.matmul_precision)
    if config.trainer.suppress_accumulate_grad_stream_mismatch_warning:
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    pl.seed_everything(config.seed, workers=True)

    data_module = build_sft_data_module(config)
    data_module.setup("fit")
    model = build_sft_lightning_module(config, data_module=data_module)
    trainer = build_sft_trainer(config)
    trainer.fit(model=model, datamodule=data_module)

    checkpoint_path = None
    if trainer.is_global_zero:
        checkpoint_path = model.save_best_decoder_checkpoint(
            config.checkpoint.dirpath,
            config.checkpoint.filename,
        )
    model.restore_best_weights()
    trainer.test(model=model, datamodule=data_module)

    return SFTExperimentResult(
        history=model.history,
        is_global_zero=trainer.is_global_zero,
        checkpoint_path=checkpoint_path,
        test_metrics=model.test_metrics,
    )
