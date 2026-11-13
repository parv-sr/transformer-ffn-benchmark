"""Exp 04: six paired Forest Cover seeds; SiLU, SwiGLU, and XGBoost.

Run from repository root:
    python -u -m experiments.exp04.exp04 | tee exp04_run.log
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from experiments.exp01.train import choose_device, num_parameters, set_seed, synchronize_shared_weights
from experiments.exp03.exp03 import atomic_csv, fit, read_csv_rows
from experiments.exp03.exp03_data import find_source, load_dataset
from experiments.exp03.exp03_model import MixedBinaryTransformer
from experiments.exp04.exp04_plots import regenerate_plots
from metrics import compute_fraud_metrics, validation_f1_threshold

MODELS = ("silu", "swiglu", "xgboost")
SEEDS = (42, 43, 44, 45, 46, 47)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/exp03"))
    p.add_argument("--output-dir", type=Path, default=Path("results/exp04"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--sample-seed", type=int, default=42, help="Fixed row sample across split seeds")
    p.add_argument("--max-rows", type=int, default=300_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--xgb-jobs", type=int, default=4)
    p.add_argument("--xgb-estimators", type=int, default=800)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.max_rows < 100
            or args.epochs < 1 or args.patience < 0 or args.batch_size < 1 or args.accum_steps < 1
            or args.lr <= 0 or args.weight_decay < 0 or args.workers < 0
            or args.xgb_jobs < 1 or args.xgb_estimators < 2):
        p.error("Invalid seeds, batch, epoch, optimizer, or XGBoost settings")
    return args


def xgb_inputs(data):
    """Use consistent categorical category domains across all three partitions."""
    features = []
    for dataset in (data.train, data.val, data.test):
        xnum, xcat, target = dataset.tensors
        frame = pd.DataFrame(xnum.numpy(), columns=list(data.numeric_columns))
        for j, name in enumerate(data.category_columns):
            frame[name] = pd.Categorical(xcat[:, j].numpy(), categories=range(data.cardinalities[j] + 1))
        features.append((frame, target.numpy()))
    return features


def fit_xgb(data, args, seed):
    train, val, test = xgb_inputs(data)
    estimator = XGBClassifier(
        n_estimators=args.xgb_estimators, max_depth=8, learning_rate=0.05,
        min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, tree_method="hist", device=args.xgb_device,
        enable_categorical=True, max_cat_to_onehot=8,
        n_jobs=args.xgb_jobs, random_state=seed,
        eval_metric=["logloss", "aucpr"], early_stopping_rounds=60,
        objective="binary:logistic",
    )
    started = time.perf_counter()
    estimator.fit(train[0], train[1], eval_set=[train, val], verbose=False)
    elapsed = time.perf_counter() - started
    # XGBoost uses its selected best_iteration for predict_proba by default.
    p_val = estimator.predict_proba(val[0])[:, 1]
    p_test = estimator.predict_proba(test[0])[:, 1]
    threshold = validation_f1_threshold(val[1], p_val)
    metrics = compute_fraud_metrics(test[1], p_test, threshold)
    history = estimator.evals_result()
    train_ce = history["validation_0"]["logloss"]
    val_ce = history["validation_1"]["logloss"]
    val_aucpr = history["validation_1"]["aucpr"]
    rows = [dict(seed=seed, model="xgboost", step=i+1, unit="round",
                 train_ce=float(train_ce[i]), val_ce=float(val_ce[i]),
                 val_ap=np.nan, val_aucpr_xgb=float(val_aucpr[i]))
            for i in range(len(val_ce))]
    result = dict(
        dataset="forest", capacity="tree", seed=seed, model="xgboost",
        parameters=np.nan, best_epoch=estimator.best_iteration+1,
        epochs_trained=len(val_ce), n_trees=estimator.best_iteration+1,
        threshold=threshold, val_ap=float(average_precision_score(val[1], p_val)),
        val_ce=float(log_loss(val[1], p_val, labels=[0, 1])),
        test_ce=float(log_loss(test[1], p_test, labels=[0, 1])),
        test_ap=metrics.average_precision, test_auc=float(roc_auc_score(test[1], p_test)),
        test_accuracy=float(accuracy_score(test[1], p_test >= threshold)),
        test_f1=metrics.f1, test_precision=metrics.precision, test_recall=metrics.recall,
        test_brier=float(brier_score_loss(test[1], p_test)),
        test_log_loss=float(log_loss(test[1], p_test, labels=[0, 1])),
        tp=metrics.tp, fp=metrics.fp, fn=metrics.fn, tn=metrics.tn,
        train_seconds=elapsed,
    )
    return rows, result


def paired_summary(metrics: list[dict], path: Path):
    if not metrics:
        return
    df = pd.DataFrame(metrics)
    cols = ["test_ap", "test_auc", "test_f1", "test_accuracy", "test_log_loss",
            "test_brier", "train_seconds"]
    table = df.groupby("model", sort=False)[cols].agg(["mean", "std"])
    table.columns = [f"{col}_{stat}" for col, stat in table.columns]
    atomic_csv(path, table.reset_index().to_dict("records"))


def run():
    args = parse_args()
    source = find_source(args.data_root, "forest")
    if args.dry_run:
        print(f"Forest source: {source}")
        print(f"Fixed {args.max_rows:,}-row sample seed {args.sample_seed}; splits {args.seeds}.")
        print(f"Models: {MODELS}; {3 * len(args.seeds)} fits; output: {args.output_dir}")
        return
    config = {
        "experiment": "exp04", "dataset": "forest (Cover_Type=1 vs 2)",
        "dataset_path": str(source.resolve()), "dataset_bytes": source.stat().st_size,
        "sample_seed": args.sample_seed, "max_rows": args.max_rows, "seeds": args.seeds,
        "models": list(MODELS), "transformer": {"d_model": 96, "depth": 4, "heads": 4,
              "mlp": "SiLU", "gated": "SwiGLU", "dropout": 0, "epochs": args.epochs,
              "patience": args.patience, "batch_size": args.batch_size,
              "accum_steps": args.accum_steps, "lr": args.lr,
              "weight_decay": args.weight_decay, "device": args.device},
        "xgboost": {"n_estimators": args.xgb_estimators, "max_depth": 8,
              "learning_rate": 0.05, "min_child_weight": 2, "subsample": 0.9,
              "colsample_bytree": 0.9, "reg_lambda": 1.0, "max_cat_to_onehot": 8,
              "early_stopping_rounds": 60, "eval_metric": ["logloss", "aucpr"],
              "device": args.xgb_device, "n_jobs": args.xgb_jobs},
        "split": "fixed row sample; stratified 70/15/15 per split seed",
        "preprocessing": "numerical train-only median/scaler, categorical train-only vocabulary",
        "selection": "Transformer validation AP; XGBoost validation aucpr; F1 threshold on validation",
        "note": "test sets overlap across seeds; boosting rounds are not Transformer epochs",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("Existing Exp 04 configuration differs: choose a new --output-dir.")
    else:
        if any((args.output_dir / fname).exists() for fname in ("metrics.csv", "history.csv")):
            raise RuntimeError("Unconfigured results already exist; refusing to mix experiments.")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    metrics_path, history_path = args.output_dir / "metrics.csv", args.output_dir / "history.csv"
    metrics, history = read_csv_rows(metrics_path), read_csv_rows(history_path)
    completed = {(int(r["seed"]), r["model"]) for r in metrics}
    regenerate_plots(history_path, args.output_dir)
    device = choose_device(args.device)

    for seed in args.seeds:
        missing = [model for model in MODELS if (seed, model) not in completed]
        if not missing:
            print(f"SKIP seed {seed}: all models complete", flush=True)
            continue
        print(f"\n=== Forest split seed {seed}, pending: {missing} ===", flush=True)
        data = load_dataset(args.data_root, "forest", args.max_rows,
                            seed=seed, sample_seed=args.sample_seed)
        print(f"Rows: {data.sampled_rows:,}; train/val/test: "
              f"{len(data.train):,}/{len(data.val):,}/{len(data.test):,}", flush=True)
        common = dict(num_numeric=len(data.numeric_columns),
                      cat_cardinalities=data.cardinalities,
                      d_model=96, depth=4, num_heads=4)
        set_seed(seed)
        silu = MixedBinaryTransformer(**common, kind="silu")
        set_seed(seed)
        swiglu = MixedBinaryTransformer(**common, kind="swiglu")
        synchronize_shared_weights(silu, swiglu)

        for kind, model in (("silu", silu), ("swiglu", swiglu), ("xgboost", None)):
            key = (seed, kind)
            if key in completed:
                continue
            if kind == "xgboost":
                run_history, result = fit_xgb(data, args, seed)
            else:
                set_seed(seed)
                # Reuse Exp03 training protocol for an exact architectural comparison.
                args.seed = seed
                start = time.perf_counter()
                run_history, result = fit(model, data, device, args, "forest", "baseline", kind)
                result["train_seconds"] = time.perf_counter() - start
                result["n_trees"] = np.nan
                run_history = [dict(seed=seed, model=kind, step=int(r["epoch"]), unit="epoch",
                                    train_ce=r["train_ce"], val_ce=r["val_ce"],
                                    val_ap=r["val_ap"], val_aucpr_xgb=np.nan)
                               for r in run_history]
            result["sample_rows"] = data.sampled_rows
            result["split_train_rows"] = len(data.train)
            result["split_test_rows"] = len(data.test)
            history = [r for r in history if (int(r["seed"]), r["model"]) != key]
            history.extend(run_history)
            atomic_csv(history_path, history)
            metrics.append(result)
            atomic_csv(metrics_path, metrics)
            completed.add(key)
            paired_summary(metrics, args.output_dir / "summary.csv")
            regenerate_plots(history_path, args.output_dir)
            print(f"  DONE {kind}/{seed}: AP={result['test_ap']:.5f} "
                  f"F1={result['test_f1']:.5f} "
                  f"seconds={result['train_seconds']:.1f}", flush=True)
        del silu, swiglu, data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nExp 04 complete. Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    run()
