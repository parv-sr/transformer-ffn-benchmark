"""Train and compare GELU-MLP vs SwiGLU encoder Transformers on creditcard.csv.

Examples:
    python train.py --csv data/creditcard.csv
    python train.py --csv data/creditcard.csv --matching equal_width --seeds 42 43 44
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dataloaders import FraudLoaders, build_dataloaders
from dataset import FEATURE_COLUMNS, load_fraud_splits
from metrics import compute_fraud_metrics, validation_f1_threshold
from models import GELUTransformer, SwiGLUTransformer
from plotting import plot_loss_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=7, help="Early stop after this many epochs without val AP improvement; 0 disables")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Run paired comparisons on one or more random seeds")
    parser.add_argument("--mlp-activation", choices=("gelu", "silu"), default="gelu",
                        help="Activation for the standard, ungated MLP")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--matching", choices=("equal_params", "equal_width"), default="equal_params",
        help="equal_params uses ~2/3 SwiGLU hidden width; equal_width uses identical hidden widths",
    )
    parser.add_argument(
        "--class-weight", choices=("sqrt", "balanced", "none"), default="sqrt",
        help="Fraud weight: sqrt(neg/pos), neg/pos, or 1.0",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--keep-duplicates", action="store_true", help="Disable duplicate-feature removal (not recommended)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1:
        parser.error("epochs, batch-size and accum-steps must be positive")
    if args.d_model % args.heads != 0 or args.depth < 1 or args.mlp_ratio <= 0:
        parser.error("d-model must be divisible by heads, depth >= 1, mlp-ratio > 0")
    if not 0 <= args.warmup_fraction < 1 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("warmup-fraction must be in [0,1); min-lr-ratio in [0,1]")
    if args.clip_grad_norm <= 0 or args.lr <= 0 or args.workers < 0:
        parser.error("clip-grad-norm and lr must be > 0; workers must be >= 0")
    return args


def choose_device(request: str) -> torch.device:
    if request != "auto":
        device = torch.device(request)
        if request == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but no CUDA device is available")
        if request == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested, but no MPS device is available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def num_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def synchronize_shared_weights(source: nn.Module, target: nn.Module) -> int:
    """Match feature tokenizer, attention, norms, and classifier initialization.

    FFNs intentionally remain independently initialized because their weight
    matrices have different shapes/semantics.
    """
    source_state, target_state = source.state_dict(), target.state_dict()
    copies = 0
    with torch.no_grad():
        for name, target_tensor in target_state.items():
            if ".ffn." in name:
                continue
            if name in source_state and target_tensor.shape == source_state[name].shape:
                target_tensor.copy_(source_state[name])
                copies += 1
    target.load_state_dict(target_state)
    return copies


def make_class_weights(y_train: torch.Tensor, strategy: str, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y_train, minlength=2).float()
    if counts.min() <= 0:
        raise ValueError("Training split must contain both classes")
    ratio = (counts[0] / counts[1]).item()
    pos_weight = {"none": 1.0, "sqrt": math.sqrt(ratio), "balanced": ratio}[strategy]
    print(f"  Training examples: normal={int(counts[0]):,}, fraud={int(counts[1]):,}; positive loss weight={pos_weight:.3f}")
    return torch.tensor([1.0, pos_weight], device=device)


def build_scheduler(optimizer: AdamW, total_updates: int, warmup_fraction: float, min_lr_ratio: float) -> LambdaLR:
    """Optimizer-step (not minibatch) warmup followed by cosine decay."""
    warmup_updates = min(total_updates - 1, round(total_updates * warmup_fraction))

    def lr_multiplier(update: int) -> float:
        if warmup_updates > 0 and update < warmup_updates:
            return (update + 1) / warmup_updates
        progress = (update - warmup_updates) / max(1, total_updates - 1 - warmup_updates)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_multiplier)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    weights: torch.Tensor,
    device: torch.device,
    accum_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss_sum, weight_sum = 0.0, 0.0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        # Same objective for both networks. The normalization in a microbatch
        # does not depend on the number of fraud examples elsewhere in the epoch.
        per_sample_ce = F.cross_entropy(logits, y, reduction="none")
        sample_weights = weights[y]
        weighted_ce = (per_sample_ce * sample_weights).sum() / sample_weights.sum()

        # The final accumulation window may contain fewer microbatches.
        # Scale using its TRUE length to avoid downweighting its gradients.
        window_start = (batch_idx // accum_steps) * accum_steps
        window_size = min(accum_steps, len(loader) - window_start)
        (weighted_ce / window_size).backward()

        weighted_loss_sum += float((per_sample_ce.detach() * sample_weights).sum().item())
        weight_sum += float(sample_weights.sum().item())

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return weighted_loss_sum / weight_sum


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return weighted CE, labels, fraud probabilities, and average precision."""
    model.eval()
    weighted_loss_sum, weight_sum = 0.0, 0.0
    targets, fraud_probs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        sample_losses = F.cross_entropy(logits, y, reduction="none")
        sample_weights = weights[y]
        weighted_loss_sum += float((sample_losses * sample_weights).sum().item())
        weight_sum += float(sample_weights.sum().item())
        fraud_probs.append(logits.softmax(dim=-1)[:, 1].cpu().numpy())
        targets.append(y.cpu().numpy())

    y_true = np.concatenate(targets)
    p_fraud = np.concatenate(fraud_probs)
    ap = float(average_precision_score(y_true, p_fraud))
    return weighted_loss_sum / weight_sum, y_true, p_fraud, ap


def train_model(
    *,
    model: nn.Module,
    name: str,
    args: argparse.Namespace,
    seed: int,
    loaders: FraudLoaders,
    weights: torch.Tensor,
    device: torch.device,
    seed_dir: Path,
    ffn_width: int,
) -> tuple[list[dict], dict]:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loaders.train) / args.accum_steps)
    scheduler = build_scheduler(
        optimizer, updates_per_epoch * args.epochs, args.warmup_fraction, args.min_lr_ratio
    )
    best_ap = -math.inf
    best_epoch = 0
    unimproved = 0
    checkpoint_path = seed_dir / f"{name}_best.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model, loaders.train, optimizer, scheduler, weights, device,
            args.accum_steps, args.clip_grad_norm,
        )
        val_loss, _, _, val_ap = evaluate(model, loaders.val, weights, device)
        history.append({
            "seed": seed, "model": name, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_average_precision": val_ap,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        elapsed = time.perf_counter() - start
        print(
            f"    [{name:6s}] epoch {epoch:02d}/{args.epochs} "
            f"train CE={train_loss:.5f}, val CE={val_loss:.5f}, "
            f"val AP={val_ap:.5f}, lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"{elapsed:.1f}s",
            flush=True,
        )
        if val_ap > best_ap:
            best_ap, best_epoch, unimproved = val_ap, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": name,
                "num_features": len(FEATURE_COLUMNS),
                "d_model": args.d_model,
                "depth": args.depth,
                "heads": args.heads,
                "ffn_hidden_dim": ffn_width,
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_average_precision": best_ap,
            }, checkpoint_path)
        else:
            unimproved += 1
            if args.patience > 0 and unimproved >= args.patience:
                print(f"    Early stopping {name} (best epoch {best_epoch}).")
                break

    # Select an epoch with VALIDATION AP. Tune threshold on that best
    # checkpoint's validation predictions, then measure on test exactly once.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_val_loss, y_val, p_val, best_val_ap = evaluate(model, loaders.val, weights, device)
    threshold = validation_f1_threshold(y_val, p_val)
    test_loss, y_test, p_test, _ = evaluate(model, loaders.test, weights, device)
    metrics = compute_fraud_metrics(y_test, p_test, threshold)

    np.savez_compressed(
        seed_dir / f"{name}_test_predictions.npz",
        y_true=y_test, p_fraud=p_test, threshold=np.asarray(threshold),
    )
    result = {
        "seed": seed,
        "model": name,
        "comparison": args.matching,
        "parameters": num_parameters(model),
        "ffn_width": ffn_width,
        "best_epoch": best_epoch,
        "val_average_precision": best_val_ap,
        "val_loss": best_val_loss,
        "threshold": threshold,
        "test_loss": test_loss,
        "test_average_precision": metrics.average_precision,
        **{f"test_{key}": value for key, value in asdict(metrics).items() if key != "average_precision"},
    }
    with (seed_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"  -> {name.upper()}: best epoch={best_epoch}, val threshold={threshold:.6f}, "
        f"TEST precision={metrics.precision:.4f}, recall={metrics.recall:.4f}, "
        f"F1={metrics.f1:.4f}, AP={metrics.average_precision:.4f}, "
        f"(TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}, TN={metrics.tn})\n",
        flush=True,
    )
    return history, result


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "configuration.json").write_text(
        json.dumps({**vars(args), "csv": str(args.csv), "output_dir": str(args.output_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"Device: {device}; no dropout, no causal attention mask")
    print(f"Comparison mode: {args.matching}; MLP activation: {args.mlp_activation}; seeds: {args.seeds}")
    all_history, all_results = [], []
    mlp_width = max(1, round(args.d_model * args.mlp_ratio))
    swiglu_width = (
        max(1, round(2 * mlp_width / 3)) if args.matching == "equal_params" else mlp_width
    )

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        set_seed(seed)
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        splits = load_fraud_splits(args.csv, seed=seed, drop_duplicates=not args.keep_duplicates)
        print(f"  Rows: {splits.num_rows_original:,} -> {splits.num_rows_after_dedup:,} after deduplication")
        for label, ds in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            print(f"  {label:5s}: {len(ds):,} examples; fraud={int(ds.targets.sum()):,}")
        weights = make_class_weights(splits.train.targets, args.class_weight, device)
        # Reuse same splits and scaler; reconstruct loaders to reset shuffle RNG.
        # Restore seed before constructing each model; additionally sync every
        # shared tensor to ensure identical initial non-FFN weights.
        set_seed(seed)
        gelu = GELUTransformer(
            num_features=len(FEATURE_COLUMNS), d_model=args.d_model,
            depth=args.depth, num_heads=args.heads, ffn_hidden_dim=mlp_width,
            mlp_activation=args.mlp_activation,
        )
        set_seed(seed)
        swiglu = SwiGLUTransformer(
            num_features=len(FEATURE_COLUMNS), d_model=args.d_model,
            depth=args.depth, num_heads=args.heads, ffn_hidden_dim=swiglu_width,
        )
        copied = synchronize_shared_weights(gelu, swiglu)
        p_gelu, p_swiglu = num_parameters(gelu), num_parameters(swiglu)
        print(
            f"  GELU hidden={mlp_width} params={p_gelu:,}; "
            f"SwiGLU hidden={swiglu_width} params={p_swiglu:,}; "
            f"gap={abs(p_gelu-p_swiglu)/p_gelu:.3%}; "
            f"shared tensors synchronized={copied}",
            flush=True,
        )
        for name, model, width in (
            (args.mlp_activation, gelu, mlp_width), ("swiglu", swiglu, swiglu_width),
        ):
            set_seed(seed)
            loaders = build_dataloaders(
                splits, batch_size=args.batch_size, seed=seed,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            history, result = train_model(
                model=model, name=name, args=args, seed=seed, loaders=loaders,
                weights=weights, device=device, seed_dir=seed_dir, ffn_width=width,
            )
            all_history.extend(history)
            all_results.append(result)

        history_df = pd.DataFrame(all_history)
        plot_loss_curves(history_df, seed_dir, seed)

    history_df = pd.DataFrame(all_history)
    results_df = pd.DataFrame(all_results)
    history_df.to_csv(args.output_dir / "history.csv", index=False)
    results_df.to_csv(args.output_dir / "metrics.csv", index=False)

    cols = ["seed", "model", "parameters", "best_epoch", "threshold", "test_precision", "test_recall", "test_f1", "test_average_precision"]
    print("\n=== HELD-OUT TEST RESULTS (fraud = positive class) ===")
    print(results_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(args.seeds) > 1:
        print("\n=== ACROSS-SEED MEAN +/- STD ===")
        keys = ["test_precision", "test_recall", "test_f1", "test_average_precision"]
        print(results_df.groupby("model")[keys].agg(["mean", "std"]).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved loss curves, checkpoints and metrics in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
