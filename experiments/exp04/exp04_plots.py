"""Exp 04: a compact three-panel loss plot, reproducible from history.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = ("silu", "swiglu", "xgboost")
TITLES = ("SiLU MLP Transformer", "SwiGLU Transformer", "XGBoost")


def regenerate_plots(history_path: Path, output_dir: Path) -> Path | None:
    if not Path(history_path).is_file():
        return None
    history = pd.read_csv(history_path)
    if history.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, kind, title in zip(axes, MODELS, TITLES):
        group = history.loc[history["model"] == kind].copy()
        ax.set_title(title)
        ax.set_xlabel("Boosting round" if kind == "xgboost" else "Epoch")
        ax.set_ylabel("Binary cross-entropy (log loss)")
        ax.grid(alpha=0.2)
        if group.empty:
            ax.text(0.5, 0.5, "Pending", ha="center", va="center", transform=ax.transAxes)
            continue
        # Individual seed traces, deliberately faint; bold lines are seed-wise
        # means at each epoch/round (partial overlap for early-stopped XGBoost).
        for _, subset in group.groupby("seed"):
            subset = subset.sort_values("step")
            ax.plot(subset["step"], subset["train_ce"], color="tab:blue", alpha=0.16, lw=0.9)
            ax.plot(subset["step"], subset["val_ce"], color="tab:orange", alpha=0.16, lw=0.9)
        avg = group.groupby("step", as_index=False)[["train_ce", "val_ce"]].mean()
        ax.plot(avg["step"], avg["train_ce"], color="tab:blue", lw=2.0, label="Train mean")
        ax.plot(avg["step"], avg["val_ce"], color="tab:orange", lw=2.0, label="Validation mean")
        ax.legend(loc="best")
        if kind != "xgboost":
            ax.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle("Exp 04 — Forest Cover | Six paired splits", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "loss_curves.png"
    temp = output_dir / ".loss_curves.tmp.png"
    try:
        fig.savefig(temp, dpi=180, bbox_inches="tight")
        temp.replace(outfile)
    finally:
        plt.close(fig)
        temp.unlink(missing_ok=True)
    return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/exp04"))
    args = parser.parse_args()
    path = regenerate_plots(args.results_dir / "history.csv", args.results_dir)
    if path is None:
        raise SystemExit(f"No history found in {args.results_dir}")
    print(f"Saved {path}")
