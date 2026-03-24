from typing import Tuple

from datamodule import SudokuDataConfig, SudokuDataModule
from models import Encoder, Predictor, TransformerConfig
from ssp import ThreeAxisSSP, ThreeAxisSSPConfig
from trainermodule import SudokuTrainer, TrainConfig


def build_components() -> Tuple[str, ThreeAxisSSP, Encoder, Predictor]:
    device = "cpu"

    embedding = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=42)).to(device)

    transformer_config = TransformerConfig(
        context_size=81,
        n_heads=4,
        head_dim=64,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
    )

    encoder = Encoder(transformer_config, embedding=embedding).to(device)
    predictor = Predictor(transformer_config, embedding=embedding).to(device)

    return device, embedding, encoder, predictor


def build_data_module(num_samples: int, seed: int) -> SudokuDataModule:
    data_config = SudokuDataConfig(
        num_samples=num_samples,
        num_cells_to_mask=1,
        seed=seed,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return SudokuDataModule(data_config)


def main() -> None:
    device, embedding, encoder, predictor = build_components()
    train_module = build_data_module(num_samples=2, seed=42)
    val_module = build_data_module(num_samples=1, seed=43)

    trainer = SudokuTrainer(
        encoder=encoder,
        predictor=predictor,
        data_module=train_module,
        val_data_module=val_module,
        embedding=embedding,
        config=TrainConfig(
            max_epochs=50,
            patience=5,
            learning_rate=1e-3,
            min_delta=1e-5,
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
