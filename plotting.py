"""Artifacts generated automatically after training."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves(history: pd.DataFrame, output_dir: Path, seed: int) -> None:
    """Each Transformer gets its own subplot with train and validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    baselines = [name for name in history["model"].unique() if name != "swiglu"]
    baseline = baselines[0] if baselines else "gelu"
    for ax, name in zip(axes, (baseline, "swiglu")):
        subset = history.loc[(history["seed"] == seed) & (history["model"] == name)]
        ax.plot(subset["epoch"], subset["train_loss"], label="Train weighted CE")
        ax.plot(subset["epoch"], subset["val_loss"], label="Validation weighted CE")
        ax.set(title=f"{name.upper()} FFN", xlabel="Epoch", ylabel="Weighted cross-entropy")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"Transformer FFN comparison | seed {seed}")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
