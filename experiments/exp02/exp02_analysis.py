"""Combine Experiment 02 results and quantify validation-loss deterioration."""
from pathlib import Path
import pandas as pd

ORDER = ("nano", "small", "baseline", "large")


def aggregate_exp2(output_dir: Path):
    root = Path(output_dir)
    metrics_frames, history_frames, rows = [], [], []
    for capacity in ORDER:
        m, h = root/capacity/"metrics.csv", root/capacity/"history.csv"
        if not m.is_file() or not h.is_file():
            continue
        metrics, history = pd.read_csv(m), pd.read_csv(h)
        metrics["capacity"] = capacity
        history["capacity"] = capacity
        metrics_frames.append(metrics)
        history_frames.append(history)
        for (seed, model), group in history.groupby(["seed", "model"]):
            group = group.sort_values("epoch")
            best_ce = group.loc[group["val_loss"].idxmin()]
            best_ap = group.loc[group["val_average_precision"].idxmax()]
            last = group.iloc[-1]
            rows.append({
                "capacity": capacity, "seed": int(seed), "model": model,
                "epoch_min_val_ce": int(best_ce["epoch"]),
                "epoch_best_val_ap": int(best_ap["epoch"]),
                "epochs_trained": int(last["epoch"]),
                "min_val_ce": float(best_ce["val_loss"]),
                "last_val_ce": float(last["val_loss"]),
                "val_ce_increase_last_minus_min": float(last["val_loss"]-best_ce["val_loss"]),
                "train_val_ce_gap_last": float(last["val_loss"]-last["train_loss"]),
            })
    if not metrics_frames:
        return
    m, h = pd.concat(metrics_frames, ignore_index=True), pd.concat(history_frames, ignore_index=True)
    m.to_csv(root/"seed_metrics.csv", index=False)
    h.to_csv(root/"epoch_history.csv", index=False)
    pd.DataFrame(rows).to_csv(root/"overfitting_summary.csv", index=False)
    cols = ["parameters", "test_precision", "test_recall", "test_f1",
            "test_average_precision", "test_loss", "best_epoch"]
    agg = m.groupby(["capacity", "model"], sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{name}_{stat}" for name, stat in agg.columns]
    agg.reset_index().to_csv(root/"capacity_summary.csv", index=False)
    print("\n=== CAPACITY SUMMARY ===", flush=True)
    print(m.groupby(["capacity", "model"], sort=False)[["test_f1", "test_average_precision"]]
          .mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("exp2"))
    aggregate_exp2(p.parse_args().output_dir)
