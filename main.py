from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from experiments import run_training_experiment


@hydra_main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    history = run_training_experiment(config)
    final_train_loss, final_val_loss = history[-1]
    print(f"Training completed. Final train loss: {final_train_loss:.6f}")
    if final_val_loss is not None:
        print(f"Final val loss: {final_val_loss:.6f}")
    print(f"Epochs run: {len(history)}")


if __name__ == "__main__":
    main()
