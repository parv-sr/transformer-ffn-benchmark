"""Three compact Exp 03 PNGs: one 4x2 loss-curve grid per dataset.

Regenerate existing plots without loading models or retraining:
    python -m experiments.exp03.exp03_plots --results-dir results/exp03
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.exp02.exp02 import CAPACITIES
from experiments.exp03.exp03_data import DATASETS


def plot_dataset_loss_curves(history: pd.DataFrame, results_dir: Path, dataset: str) -> Path | None:
    """Plot all four capacities x two FFNs, leaving unfinished runs visibly blank."""
    results_dir = Path(results_dir)
    subset = history.loc[history["dataset"] == dataset] if "dataset" in history else pd.DataFrame()
    if subset.empty:
        return None

    fig, axes = plt.subplots(len(CAPACITIES), 2, figsize=(12, 13), sharex=True)
    fig.suptitle(f"Exp 03 — {dataset.title()} | Training vs validation CE", fontsize=16)
    legends = None
    for i, capacity in enumerate(CAPACITIES):
        for j, model in enumerate(("silu", "swiglu")):
            ax = axes[i, j]
            curve = subset.loc[
                (subset["capacity"] == capacity) & (subset["model"] == model)
            ].sort_values("epoch")
            ax.set_title(f"{capacity.title()} · {'SiLU MLP' if model == 'silu' else 'SwiGLU'}")
            if curve.empty:
                ax.text(0.5, 0.5, "Pending", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_axis_off()
                continue
            a, = ax.plot(curve["epoch"], curve["train_ce"], label="Train CE", lw=1.8)
            b, = ax.plot(curve["epoch"], curve["val_ce"], label="Validation CE", lw=1.8)
            legends = (a, b)
            ax.grid(alpha=0.25)
            ax.set_ylabel("Cross-entropy")
            ax.set_xlabel("Epoch")
            ax.set_xlim(left=1)
            ax.xaxis.get_major_locator().set_params(integer=True)
    if legends:
        fig.legend(legends, ["Train CE", "Validation CE"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    output = results_dir / f"{dataset}_loss_curves.png"
    temp = results_dir / f".{dataset}_loss_curves.tmp.png"
    try:
        fig.savefig(temp, dpi=150, bbox_inches="tight")
        temp.replace(output)
    finally:
        plt.close(fig)
        temp.unlink(missing_ok=True)
    return output


def regenerate_plots(history_path: Path, results_dir: Path) -> list[Path]:
    if not Path(history_path).is_file():
        return []
    history = pd.read_csv(history_path)
    if history.empty:
        return []
    paths = [plot_dataset_loss_curves(history, results_dir, dataset) for dataset in DATASETS]
    return [path for path in paths if path is not None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/exp03"))
    args = parser.parse_args()
    paths = regenerate_plots(args.results_dir / "history.csv", args.results_dir)
    if not paths:
        raise SystemExit(f"No history found at {args.results_dir / 'history.csv'}")
    for path in paths:
        print(f"Saved {path}")
