from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from datamodule import (
    LinearMaskCurriculum,
    SudokuDataConfig,
    SudokuDataModule,
)
from models import Encoder, Predictor, SudokuRepresentation, TransformerConfig
from trainermodule import EvalMetrics, SudokuTrainer, TrainConfig

ONE_BATCH_OVERFIT_OVERRIDES: dict[str, Any] = {
    "data": {
        "num_samples": 24,
        "num_cells_to_mask": 2,
        "seed": 123,
        "unique_solution": False,
        "randomize_mask_per_access": False,
        "batch_size": 24,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": True,
    },
    "validation": {
        "enabled": False,
        "num_samples": 1,
        "num_cells_to_mask": 18,
        "seed": 43,
        "unique_solution": False,
        "batch_size": 1,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": False,
    },
    "curriculum": {
        "enabled": True,
        "mode": "adaptive",
        "max_mask": 64,
        "num_epochs": 20,
        "step": 1,
        "patience": 32,
        "min_delta": 1.0e-6,
    },
    "training": {
        "max_epochs": 2000,
        "patience": 1000,
        "learning_rate": 1e-4,
        "prototype_logit_scale": 10.0,
        "min_delta": 1e-6,
        "device": "cpu",
    },
}


@dataclass(frozen=True)
class ExperimentResult:
    history: list[tuple[float, float | None]]
    validation_metrics: EvalMetrics | None = None
    test_metrics: EvalMetrics | None = None


def build_data_module(data_config: DictConfig, curriculum_config: DictConfig) -> SudokuDataModule:
    curriculum = None
    if curriculum_config.enabled and curriculum_config.mode == "linear":
        curriculum = LinearMaskCurriculum(
            start=data_config.num_cells_to_mask,
            max_mask=curriculum_config.max_mask,
            num_epochs=curriculum_config.num_epochs,
        )

    if curriculum_config.enabled and curriculum_config.mode not in {"linear", "adaptive"}:
        raise ValueError(
            f"Unsupported curriculum.mode={curriculum_config.mode}. "
            "Use 'linear' or 'adaptive'."
        )

    resolved_data_config = SudokuDataConfig(
        num_samples=data_config.num_samples,
        num_cells_to_mask=data_config.num_cells_to_mask,
        seed=data_config.seed,
        unique_solution=data_config.unique_solution,
        randomize_mask_per_access=data_config.randomize_mask_per_access,
        mask_cells_curriculum=curriculum,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last,
    )
    if resolved_data_config.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    return SudokuDataModule(resolved_data_config)


def build_components(
    config: DictConfig,
) -> tuple[str, SudokuRepresentation, Encoder, Predictor]:
    device = config.training.device

    transformer_config = TransformerConfig(
        context_size=config.model.context_size,
        n_heads=config.model.n_heads,
        head_dim=config.model.head_dim,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
    )
    if config.ssp.dim != transformer_config.d_model:
        raise ValueError(
            f"ssp.dim ({config.ssp.dim}) must match model d_model ({transformer_config.d_model})."
        )

    representation = SudokuRepresentation(
        d_model=transformer_config.d_model,
        seed=config.ssp.seed,
    ).to(device)
    encoder = Encoder(transformer_config, representation=representation).to(device)
    predictor = Predictor(transformer_config, representation=representation).to(device)
    return device, representation, encoder, predictor


def build_train_config(config: DictConfig, device: str) -> TrainConfig:
    return TrainConfig(
        max_epochs=config.training.max_epochs,
        patience=config.training.patience,
        learning_rate=config.training.learning_rate,
        prototype_logit_scale=config.training.prototype_logit_scale,
        min_delta=config.training.min_delta,
        device=device,
        curriculum_enabled=config.curriculum.enabled,
        curriculum_mode=config.curriculum.mode,
        curriculum_step=config.curriculum.step,
        curriculum_patience=config.curriculum.patience,
        curriculum_min_delta=config.curriculum.min_delta,
        curriculum_max_mask=config.curriculum.max_mask,
        curriculum_num_epochs=config.curriculum.num_epochs,
    )


def create_trainer(config: DictConfig) -> SudokuTrainer:
    train_data = build_data_module(config.data, config.curriculum)
    val_data = (
        build_data_module(config.validation, config.curriculum)
        if config.validation.enabled
        else None
    )
    test_data = (
        build_data_module(config.test, config.curriculum)
        if config.test.enabled
        else None
    )

    device, representation, encoder, predictor = build_components(config)
    return SudokuTrainer(
        encoder=encoder,
        predictor=predictor,
        data_module=train_data,
        val_data_module=val_data,
        test_data_module=test_data,
        representation=representation,
        config=build_train_config(config, device),
    )


def run_training_experiment(config: DictConfig) -> ExperimentResult:
    trainer = create_trainer(config)
    history = trainer.train()
    validation_metrics = (
        trainer.evaluate("validation") if config.validation.enabled else None
    )
    test_metrics = trainer.evaluate("test") if config.test.enabled else None
    return ExperimentResult(
        history=history,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def build_one_batch_overfit_config(base_config: DictConfig) -> DictConfig:
    return OmegaConf.merge(base_config, OmegaConf.create(ONE_BATCH_OVERFIT_OVERRIDES))


def run_one_batch_overfit(base_config: DictConfig) -> tuple[
    DictConfig,
    ExperimentResult,
]:
    config = build_one_batch_overfit_config(base_config)
    OmegaConf.resolve(config)
    result = run_training_experiment(config)
    return config, result
