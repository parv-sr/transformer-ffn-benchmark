"""Experiment 03: 3 datasets x 4 capacities x 2 FFNs, one seed, compact outputs.

CSV: results/exp03/{metrics,history,datasets}.csv
JSON: results/exp03/config.json
No checkpoints or prediction dumps. Completed fits are skipped on resume.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from experiments.exp02.exp02 import CAPACITIES
from experiments.exp03.exp03_data import DATASETS, find_source, load_dataset
from experiments.exp03.exp03_model import MixedBinaryTransformer
from experiments.exp03.exp03_plots import plot_dataset_loss_curves, regenerate_plots
from metrics import compute_fraud_metrics, validation_f1_threshold
from experiments.exp01.train import (build_scheduler, choose_device, num_parameters,
                   set_seed, synchronize_shared_weights)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/exp03"))
    p.add_argument("--output-dir", type=Path, default=Path("results/exp03"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=300_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.max_rows < 100 or args.batch_size < 1 or args.accum_steps < 1:
        p.error("epochs, batch-size, accum-steps must be >0; max-rows must be >=100")
    if args.patience < 0 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        p.error("invalid patience/workers/learning-rate/weight-decay")
    return args


def make_loader(data, batch_size, workers, seed, shuffle, pin_memory):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None,
                      num_workers=workers, pin_memory=pin_memory,
                      persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets, probabilities = [], []
    total_loss, count = 0.0, 0
    for num, cat, y in loader:
        logits = model(num.to(device, non_blocking=True), cat.to(device, non_blocking=True))
        y_device = y.to(device, non_blocking=True)
        total_loss += F.cross_entropy(logits, y_device, reduction="sum").item()
        count += len(y)
        targets.append(y.numpy())
        probabilities.append(logits.softmax(dim=-1)[:, 1].cpu().numpy())
    return (total_loss / count, np.concatenate(targets),
            np.concatenate(probabilities))


def fit(model, data, device, args, dataset, capacity, kind):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(data.train, args.batch_size, args.workers, args.seed, True, pin_memory)
    val_loader = make_loader(data.val, args.batch_size, args.workers, args.seed, False, pin_memory)
    test_loader = make_loader(data.test, args.batch_size, args.workers, args.seed, False, pin_memory)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    scheduler = build_scheduler(optimizer, steps_per_epoch * args.epochs, 0.05, 0.05)
    best_ap, best_state, best_epoch, no_improvement = -math.inf, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, total = 0.0, 0
        for batch_idx, (num, cat, y) in enumerate(train_loader):
            num = num.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(num, cat)
            loss = F.cross_entropy(logits, y)
            window_start = (batch_idx // args.accum_steps) * args.accum_steps
            window_size = min(args.accum_steps, len(train_loader) - window_start)
            (loss / window_size).backward()
            loss_sum += loss.item() * len(y)
            total += len(y)
            if (batch_idx + 1) % args.accum_steps == 0 or batch_idx + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_ce, y_val, p_val = evaluate(model, val_loader, device)
        val_ap = float(average_precision_score(y_val, p_val))
        history.append(dict(dataset=dataset, capacity=capacity, model=kind, seed=args.seed,
                            epoch=epoch, train_ce=loss_sum / total, val_ce=val_ce,
                            val_ap=val_ap, lr=optimizer.param_groups[0]["lr"]))
        print(f"  [{dataset}/{capacity}/{kind}] epoch {epoch:02d}: "
              f"train CE={loss_sum/total:.5f}, val CE={val_ce:.5f}, val AP={val_ap:.5f}",
              flush=True)
        if val_ap > best_ap:
            best_ap, best_epoch, no_improvement = val_ap, epoch, 0
            # One small checkpoint in RAM only; no per-run artifact proliferation.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improvement += 1
            if args.patience and no_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    val_ce, y_val, p_val = evaluate(model, val_loader, device)
    threshold = validation_f1_threshold(y_val, p_val)
    test_ce, y_test, p_test = evaluate(model, test_loader, device)
    measures = compute_fraud_metrics(y_test, p_test, threshold)
    result = dict(
        dataset=dataset, capacity=capacity, model=kind, seed=args.seed,
        parameters=num_parameters(model), epochs_trained=len(history), best_epoch=best_epoch,
        threshold=threshold, val_ap=best_ap, val_ce=val_ce,
        test_ce=test_ce, test_ap=measures.average_precision,
        test_auc=float(roc_auc_score(y_test, p_test)),
        test_accuracy=float(accuracy_score(y_test, p_test >= threshold)),
        test_f1=measures.f1, test_precision=measures.precision, test_recall=measures.recall,
        test_brier=float(brier_score_loss(y_test, p_test)),
        test_log_loss=float(log_loss(y_test, p_test, labels=[0, 1])),
        tp=measures.tp, fp=measures.fp, fn=measures.fn, tn=measures.tn,
    )
    del model, best_state, optimizer
    return history, result


def atomic_csv(path, records):
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_csv_rows(path):
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def run():
    args = args_parser()
    if args.dry_run:
        print("Experiment 03: 3 datasets x 4 scales x 2 models = 24 runs.")
        for dataset in DATASETS:
            try:
                source = find_source(args.data_root, dataset)
                print(f"  {dataset}: {source}")
            except FileNotFoundError as e:
                print(f"  {dataset}: MISSING ({e})")
        print(f"seed={args.seed}, max_rows={args.max_rows}, epochs={args.epochs}, "
              f"output={args.output_dir}, no artifacts created")
        return

    paths = {name: find_source(args.data_root, name) for name in DATASETS}
    config = dict(
        version=1, datasets=list(DATASETS), capacities=CAPACITIES, models=["silu", "swiglu"],
        seed=args.seed, max_rows=args.max_rows, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size, accum_steps=args.accum_steps,
        lr=args.lr, weight_decay=args.weight_decay, workers=args.workers,
        device=args.device, data_root=str(args.data_root.resolve()),
        sources={k: {"path": str(p.resolve()), "bytes": p.stat().st_size}
                 for k, p in paths.items()},
        split="stratified random 70/15/15; sample first; seed fixed",
        normalization="train-only median imputation + standardization",
        categorical="train-only vocab; missing/unseen=0; per-field embeddings",
        objective="unweighted binary CE; no dropout; best val AP; threshold from val F1",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Existing results/exp03 config does not match. "
                               "Choose a new --output-dir rather than mixing experiments.")
    else:
        if any((args.output_dir / name).exists()
               for name in ("metrics.csv", "history.csv", "datasets.csv")):
            raise RuntimeError("Output contains results but no config.json; refusing to mix runs")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    device = choose_device(args.device)
    metrics_path = args.output_dir / "metrics.csv"
    history_path = args.output_dir / "history.csv"
    data_path = args.output_dir / "datasets.csv"
    metrics = read_csv_rows(metrics_path)
    history = read_csv_rows(history_path)
    datasets_info = read_csv_rows(data_path)
    regenerate_plots(history_path, args.output_dir)
    completed = {(r["dataset"], r["capacity"], r["model"]) for r in metrics}

    for dataset in DATASETS:
        if all((dataset, scale, model) in completed for scale in CAPACITIES
               for model in ("silu", "swiglu")):
            print(f"SKIP completed dataset: {dataset}", flush=True)
            continue
        print(f"\n=== {dataset} (sample <= {args.max_rows:,}) ===", flush=True)
        data = load_dataset(args.data_root, dataset, args.max_rows, args.seed)
        if not any(r["dataset"] == dataset for r in datasets_info):
            datasets_info.append(dict(
                dataset=dataset, source_population_rows=data.population_rows,
                sampled_rows=data.sampled_rows, train_rows=len(data.train),
                val_rows=len(data.val), test_rows=len(data.test),
                positive_fraction=data.positive_rate,
                numeric_features=len(data.numeric_columns),
                categorical_features=len(data.category_columns),
                category_vocab_rows=sum(k + 1 for k in data.cardinalities),
            ))
            atomic_csv(data_path, datasets_info)
        print(f"  {data.sampled_rows:,} rows; positives={data.positive_rate:.3f}; "
              f"numeric={len(data.numeric_columns)}, categorical={len(data.category_columns)}",
              flush=True)
        for capacity, spec in CAPACITIES.items():
            if all((dataset, capacity, model) in completed for model in ("silu", "swiglu")):
                continue
            set_seed(args.seed)
            common = dict(num_numeric=len(data.numeric_columns),
                          cat_cardinalities=data.cardinalities,
                          d_model=spec["d_model"], depth=spec["depth"],
                          num_heads=spec["heads"])
            silu = MixedBinaryTransformer(**common, kind="silu")
            set_seed(args.seed)
            swiglu = MixedBinaryTransformer(**common, kind="swiglu")
            synchronize_shared_weights(silu, swiglu)
            print(f"  {capacity}: SiLU={num_parameters(silu):,} params; "
                  f"SwiGLU={num_parameters(swiglu):,} params", flush=True)
            for kind, model in (("silu", silu), ("swiglu", swiglu)):
                key = dataset, capacity, kind
                if key in completed:
                    print(f"  SKIP {key}", flush=True)
                    continue
                set_seed(args.seed)
                run_history, result = fit(model, data, device, args, dataset, capacity, kind)
                # On a crash after saving history but before metrics, rerun replaces it.
                history = [r for r in history if
                           (r["dataset"], r["capacity"], r["model"]) != key]
                history.extend(run_history)
                atomic_csv(history_path, history)
                metrics.append(result)
                atomic_csv(metrics_path, metrics)
                plot_dataset_loss_curves(pd.DataFrame(history), args.output_dir, dataset)
                completed.add(key)
                print(f"  DONE {key}: F1={result['test_f1']:.4f}, "
                      f"AP={result['test_ap']:.4f}", flush=True)
    print(f"\nFinished. Results in {args.output_dir}", flush=True)


if __name__ == "__main__":
    run()
