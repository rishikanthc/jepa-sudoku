from omegaconf import OmegaConf

from experiments import run_one_batch_overfit


def main() -> None:
    base_config = OmegaConf.load("configs/default.yaml")
    config, result = run_one_batch_overfit(base_config)
    history = result.history

    final_train_loss, final_val_loss = history[-1]
    print("One-batch overfit run completed.")
    print(
        "Resolved setup: "
        f"num_samples={config.data.num_samples}, "
        f"batch_size={config.data.batch_size}, "
        f"randomize_mask_per_access={config.data.randomize_mask_per_access}, "
        f"validation_enabled={config.validation.enabled}"
    )
    print(f"Final train loss: {final_train_loss:.6f}")
    if final_val_loss is not None:
        print(f"Final val loss: {final_val_loss:.6f}")
    print(f"Epochs run: {len(history)}")


if __name__ == "__main__":
    main()
