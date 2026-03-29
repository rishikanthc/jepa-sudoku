from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from jepa_sudoku.training.experiments import run_training_experiment


@hydra_main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    result = run_training_experiment(config)
    if not result.is_global_zero:
        return
    if result.history:
        print(f"Pretraining completed. Final train loss: {result.history[-1]:.6f}")
    if result.checkpoint_path:
        print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Epochs run: {len(result.history)}")


if __name__ == "__main__":
    main()
