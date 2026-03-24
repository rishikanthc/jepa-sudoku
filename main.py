from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from datamodule import (
    LinearMaskCurriculum,
    SudokuDataConfig,
    SudokuDataModule,
)
from models import Encoder, Predictor, TransformerConfig
from ssp import ThreeAxisSSP, ThreeAxisSSPConfig
from trainermodule import SudokuTrainer, TrainConfig


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


def build_components(config: DictConfig) -> tuple[str, ThreeAxisSSP, Encoder, Predictor]:
    device = config.training.device

    embedding = ThreeAxisSSP(
        ThreeAxisSSPConfig(dim=config.ssp.dim, seed=config.ssp.seed)
    ).to(device)

    transformer_config = TransformerConfig(
        context_size=config.model.context_size,
        n_heads=config.model.n_heads,
        head_dim=config.model.head_dim,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
    )

    encoder = Encoder(transformer_config, embedding=embedding).to(device)
    predictor = Predictor(transformer_config, embedding=embedding).to(device)
    return device, embedding, encoder, predictor


@hydra_main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)

    train_data = build_data_module(config.data, config.curriculum)
    val_data = (
        build_data_module(config.validation, config.curriculum)
        if config.validation.enabled
        else None
    )

    device, embedding, encoder, predictor = build_components(config)
    trainer = SudokuTrainer(
        encoder=encoder,
        predictor=predictor,
        data_module=train_data,
        val_data_module=val_data,
        embedding=embedding,
        config=TrainConfig(
            max_epochs=config.training.max_epochs,
            patience=config.training.patience,
            learning_rate=config.training.learning_rate,
            min_delta=config.training.min_delta,
            device=device,
            curriculum_enabled=config.curriculum.enabled,
            curriculum_mode=config.curriculum.mode,
            curriculum_step=config.curriculum.step,
            curriculum_patience=config.curriculum.patience,
            curriculum_min_delta=config.curriculum.min_delta,
            curriculum_max_mask=config.curriculum.max_mask,
            curriculum_num_epochs=config.curriculum.num_epochs,
        ),
    )

    history = trainer.train()
    final_train_loss, final_val_loss = history[-1]
    print(f"Training completed. Final train loss: {final_train_loss:.6f}")
    if final_val_loss is not None:
        print(f"Final val loss: {final_val_loss:.6f}")
    print(f"Epochs run: {len(history)}")


if __name__ == "__main__":
    main()
