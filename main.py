from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from experiments import run_training_experiment


@hydra_main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    result = run_training_experiment(config)
    history = result.history
    final_train_loss, final_val_loss = history[-1]
    print(f"Training completed. Final train loss: {final_train_loss:.6f}")
    if final_val_loss is not None:
        print(f"Final val loss: {final_val_loss:.6f}")
    if result.validation_metrics is not None:
        print(
            "Validation eval: "
            f"loss={result.validation_metrics.loss:.6f} "
            f"acc={result.validation_metrics.accuracy:.6f}"
        )
    if result.test_metrics is not None:
        print(
            "Test eval: "
            f"loss={result.test_metrics.loss:.6f} "
            f"acc={result.test_metrics.accuracy:.6f}"
        )
    print(f"Epochs run: {len(history)}")


if __name__ == "__main__":
    main()
