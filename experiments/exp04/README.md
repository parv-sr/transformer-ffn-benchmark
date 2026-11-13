# Exp 04 — Forest Cover: Transformers vs XGBoost

**Question:** Does SwiGLU retain an advantage across splits, and how competitive is an FT-style Transformer against a strong conventional tabular learner?

## Fixed protocol

| Component | Choice |
|---|---|
| Dataset | Forest Cover Type, original classes 1/2 only |
| Sample | 300K rows, fixed sample seed 42 |
| Split seeds | **42, 43, 44, 45, 46, 47** |
| Split | Stratified 70/15/15, different per seed; paired across all three models |
| Transformers | d_model 96, 4 layers, 4 heads, approximately 454K parameters |
| FFNs | SiLU MLP (4d hidden), SwiGLU (≈8d/3 hidden), parameter matched |
| Optimization | AdamW 3e-4, max 12 epochs, patience 3, no dropout, best validation AP |
| XGBoost | Hist trees; depth 8, up to 800 rounds, learning rate .05, subsample .9, colsample .9 |
| XGBoost stopping | 60-round validation AUCPR patience; native categorical splits |
| Decision threshold | Validation F1 optimization for each model/seed |
| Primary metric | Held-out AP; plus F1, ROC-AUC, log loss, Brier, training seconds |

Both model families use identical samples, train-fitted numerical scalers, and train-only category-ID mappings. XGBoost treats categories natively using pandas categorical columns, not arbitrary ordinal numbers. Its tree count is **not** parameter-matched to the Transformers.

## Run

Use the already downloaded `data/exp03/forest/covtype.csv` (or other uniquely named CSV in that folder), from repository root:

```bash
uv pip install -r requirements.txt
uv run python -m experiments.exp04.exp04 --dry-run
uv run python -u -m experiments.exp04.exp04 2>&1 | tee exp04_run.log
```

XGBoost uses CPU by default (4 threads). Optionally use `--xgb-device cuda` if supported by your installation.

## Outputs

```text
results/exp04/
  config.json
  metrics.csv
  history.csv
  summary.csv
  loss_curves.png
```

The PNG contains three panels (SiLU, SwiGLU, XGBoost), displaying each seed's train/validation cross-entropy and the across-seed mean. Each panel has its own x-axis: **Transformer epochs are not equivalent to XGBoost boosting rounds**.

Generate/refresh PNG without retraining:

```bash
uv run python -m experiments.exp04.exp04_plots --results-dir results/exp04
```

Completed fits are skipped on restart. Changing configuration requires a new output directory; no saved checkpoints or predictions.

## Interpretation

Compare **paired** per-seed differences (SwiGLU − SiLU, each Transformer − XGBoost). The six test partitions overlap; they are not independent datasets. XGBoost stopping uses AUCPR while the Transformer uses stepwise validation AP. This is one competitive, fixed XGBoost configuration, not an exhaustive hyperparameter search.

**Results:** pending.
