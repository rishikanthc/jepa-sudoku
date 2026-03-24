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


def build_data_module(config: DictConfig) -> SudokuDataModule:
    curriculum = None
    if config.curriculum.enabled:
        curriculum = LinearMaskCurriculum(
            start=config.curriculum.start,
            max_mask=config.curriculum.max_mask,
            num_epochs=config.curriculum.num_epochs,
        )

    data_config = SudokuDataConfig(
        num_samples=config.num_samples,
        num_cells_to_mask=config.num_cells_to_mask,
        seed=config.seed,
        unique_solution=config.unique_solution,
        mask_cells_curriculum=curriculum,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    if data_config.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    return SudokuDataModule(data_config)


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

    train_data = build_data_module(config.data)
    val_data = build_data_module(config.validation) if config.validation.enabled else None

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
