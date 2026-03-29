from omegaconf import OmegaConf

from jepa_sudoku.training.experiments import run_one_batch_overfit


def main() -> None:
    base_config = OmegaConf.load("configs/default.yaml")
    config, result = run_one_batch_overfit(base_config)
    if not result.is_global_zero:
        return
    print("One-batch overfit run completed.")
    print(
        "Resolved setup: "
        f"num_samples={config.data.num_samples}, "
        f"batch_size={config.data.batch_size}, "
        f"randomize_mask_per_access={config.data.randomize_mask_per_access}, "
        f"curriculum_enabled={config.curriculum.enabled}"
    )
    if result.history:
        print(f"Final train loss: {result.history[-1]:.6f}")
    if result.checkpoint_path:
        print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Epochs run: {len(result.history)}")


if __name__ == "__main__":
    main()
