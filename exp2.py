"""Experiment 02: four capacities, two FFNs, three paired seeds; no dropout."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from exp2_analysis import aggregate_exp2
from models import GELUTransformer, SwiGLUTransformer

CAPACITIES = {
    "nano": {"d_model": 16, "depth": 2, "heads": 2},
    "small": {"d_model": 48, "depth": 3, "heads": 4},
    "baseline": {"d_model": 96, "depth": 4, "heads": 4},
    "large": {"d_model": 128, "depth": 4, "heads": 4},
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("exp2"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    p.add_argument("--mlp-activation", choices=("gelu", "silu"), default="silu")
    p.add_argument("--only", nargs="+", choices=tuple(CAPACITIES), default=list(CAPACITIES))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing capacity results")
    args = p.parse_args()
    if not args.dry_run and not args.csv.is_file():
        p.error(f"Missing dataset: {args.csv}")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must be unique")
    return args


def main():
    args = parse_args()
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        cfg = CAPACITIES[name]
        d, depth, heads = cfg["d_model"], cfg["depth"], cfg["heads"]
        mlp = GELUTransformer(d_model=d, depth=depth, num_heads=heads,
                              ffn_hidden_dim=4*d, mlp_activation=args.mlp_activation)
        swiglu = SwiGLUTransformer(d_model=d, depth=depth, num_heads=heads,
                                    ffn_hidden_dim=round(8*d/3))
        na, nb = count_parameters(mlp), count_parameters(swiglu)
        print(f"{name:8s} MLP={na:,} SwiGLU={nb:,} gap={abs(na-nb)/na:.2%}", flush=True)
        out = args.output_dir / name
        command = [
            sys.executable, "-u", str(Path(__file__).with_name("train.py")),
            "--csv", str(args.csv), "--output-dir", str(out),
            "--d-model", str(d), "--depth", str(depth), "--heads", str(heads),
            "--matching", "equal_params", "--mlp-ratio", "4",
            "--mlp-activation", args.mlp_activation, "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--batch-size", str(args.batch_size),
            "--accum-steps", str(args.accum_steps), "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay), "--workers", str(args.workers),
            "--device", args.device, "--seeds", *map(str, args.seeds),
        ]
        requested = {
            "csv": str(args.csv), "output_dir": str(out), "d_model": d,
            "depth": depth, "heads": heads, "matching": "equal_params",
            "mlp_ratio": 4.0, "mlp_activation": args.mlp_activation,
            "epochs": args.epochs, "patience": args.patience, "seeds": args.seeds,
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "workers": args.workers, "device": args.device,
        }
        if args.dry_run:
            print("DRY RUN:", " ".join(command), flush=True)
            continue
        config = out / "configuration.json"
        metrics = out / "metrics.csv"
        history = out / "history.csv"
        if out.exists() and not args.force:
            if not config.is_file():
                raise RuntimeError(f"{out} already exists without configuration; use --force or a new output directory")
            actual = json.loads(config.read_text())
            mismatch = {key: (actual.get(key), value) for key, value in requested.items()
                        if actual.get(key) != value}
            if mismatch:
                raise RuntimeError(f"{out} configuration mismatch: {mismatch}; use --force or a new directory")
            if metrics.is_file() and history.is_file():
                rows = pd.read_csv(metrics)
                expected = {(seed, model) for seed in args.seeds
                            for model in (args.mlp_activation, "swiglu")}
                if set(zip(rows["seed"], rows["model"])) == expected:
                    print(f"SKIP complete: {out}", flush=True)
                    aggregate_exp2(args.output_dir)
                    continue
            raise RuntimeError(f"{out} is incomplete; use --force to restart that capacity")
        out.mkdir(parents=True, exist_ok=True)
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        aggregate_exp2(args.output_dir)
    if not args.dry_run:
        aggregate_exp2(args.output_dir)
        print(f"Complete: {args.output_dir}/capacity_summary.csv", flush=True)


if __name__ == "__main__":
    main()
