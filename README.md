# Credit Card Fraud: GELU-MLP vs SwiGLU Transformer

A controlled, **no-dropout** PyTorch comparison of two non-causal, encoder-only
Transformers. The **only architectural difference is the feed-forward network**:
standard GELU MLP versus SwiGLU. Both models have independent training runs, but
use identical train/val/test partitions, standardized features, minibatch order,
training hyperparameters, optimizer/scheduler, and initially identical weights
for all shared submodules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
# Must produce: data/creditcard.csv
```

If Kaggle CLI prompts for credentials, sign in to Kaggle and configure its API
credentials, or manually download and unzip the CSV from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

## Run

```bash
# Fair, approximately parameter-matched main comparison:
python train.py --csv data/creditcard.csv

# Repeat on three seeds to inspect variance:
python train.py --csv data/creditcard.csv --seeds 42 43 44

# Literal FFN replacement using the same internal hidden width, despite
# SwiGLU having ~50% more FFN matrix parameters:
python train.py --csv data/creditcard.csv --matching equal_width --output-dir results_equal_width
```

You can tune `--d-model`, `--depth`, `--heads`, `--mlp-ratio`, `--epochs`,
`--batch-size`, `--accum-steps`, `--lr`, `--patience`, and `--device`.

The defaults are `d_model=96`, 4 layers, 4 heads, batch size 512, 2 gradient
accumulation steps (effective batch up to 1024), 20 epochs, warmup + cosine
learning-rate schedule, AdamW, gradient-norm clipping at 1.0, and patience 7.
There are no dropout modules or stochastic attention masks.

## Architecture

```
Input transaction: [batch, 30]
    | feature-wise affine projection: token_j = x_j * W_j + b_j
30 tokens of width 96
    | prepend learned [CLS]
[batch, 31, 96]
    |
    +-- 4 x Pre-LN encoder block ---------------------------+
    |   x = x + MultiHeadAttention(LayerNorm(x))           |
    |       4 heads; attention across ALL tokens; no mask  |
    |   x = x + FFN(LayerNorm(x))                          |
    |       GELU MLP OR SwiGLU; no dropout                  |
    +-------------------------------------------------------+
    |
Final LayerNorm -> [CLS] representation -> Linear(96, 2)
    |
2 raw logits for CrossEntropyLoss
    |
Softmax -> p(normal), p(fraud) for prediction and evaluation
```

**GELU FFN:** `down(GELU(up(x)))`, where `up: 96 -> 384` and `down: 384 -> 96`.

**SwiGLU FFN:** `down(SiLU(gate(x)) * up(x))`, where the two first matrices are
`96 -> 256` and `down: 256 -> 96` in parameter-matched mode.

With hidden width `4*d` for GELU, a SwiGLU hidden width of `(8/3)*d` brings the
number of weight parameters close. The two biases differ slightly; exact total
parameter counts and percent difference are printed before training. For a
literal equal-hidden-width comparison, use `--matching equal_width`. The models
have otherwise identical architecture; no positional embeddings are needed
because each numerical column has a distinct feature-specific projection.

## Dataset and experiment design

- Original Kaggle `creditcard.csv`: 284,807 rows, 30 numeric input columns
  (`Time`, `V1` ... `V28`, `Amount`) and binary `Class` (fraud=1).
- **Duplicate feature vectors are removed by default** before splitting to
  prevent identical inputs appearing in different partitions. Thus the run
  may use fewer than 284,807 records. Use `--keep-duplicates` to disable.
- Stratified random split: **70% training, 15% validation, 15% test**.
  `StandardScaler` fits *only* on the training split and transforms all 30
  features. Validation/test prevalence remains natural (not artificially
  balanced). Both models use the same splits.
- Approximate imbalance handling via weighted cross-entropy. Default fraud
  weight is `sqrt(num_normal / num_fraud)`; options `balanced` or `none`.
  Data are **not** artificially oversampled, so label prevalence in validation
  and test is untouched.
- Optimizer: AdamW; warmup+cosine schedule is stepped after each **optimizer
  update**, including when using accumulation; clip gradient global norm to
  `1.0` before each optimizer step.
- Save the best checkpoint by **validation average precision (AP)** (area
  under the stepwise precision-recall curve). Choose the class decision
  threshold that maximizes **validation F1** using that checkpoint, then
  evaluate on untouched test data exactly once.
- Report class-1 test precision, recall, F1, AP, confusion-matrix counts and
  loss. Precision, recall and F1 depend on that validation-chosen threshold.
- Multiple seeds (`--seeds 42 43 44`) produce per-seed results and across-seed
  mean/std for paired model comparisons.

The split is stratified rather than chronological: results are appropriate for
**architectural comparisons on this corpus**, not claims of deployment-ready
fraud detection under future distribution shift. Duplicates removed before
splitting are deliberately reported. The score on a single dataset cannot
establish that either FFN is a universal replacement in all tasks.

## Outputs

```
results/
  configuration.json
  history.csv                      # Epoch-by-epoch train/validation losses, AP and LR
  metrics.csv                      # Final test precision, recall, F1, AP etc.
  seed_42/
    loss_curves.png                # Both models' training and validation losses
    gelu_best.pt
    gelu_metrics.json
    gelu_test_predictions.npz      # Test truth, probabilities, chosen threshold
    swiglu_best.pt
    swiglu_metrics.json
    swiglu_test_predictions.npz
```

A summary comparison table (test precision/recall/F1/AP) is printed by the
training script. Test labels are never used for checkpoint or threshold selection.

## Files

- `dataset.py`: `FraudDataset` and training-only standardization + split logic.
- `dataloaders.py`: independent train/validation/test DataLoader builders.
- `models/common.py`: shared tokenizer, pre-LN non-causal attention blocks,
  `[CLS]` classifier, and softmax inference method.
- `models/mlp_transformer.py`: the GELU model definition.
- `models/swiglu_transformer.py`: the SwiGLU model definition.
- `models/ffn.py`: both FFN formulas.
- `train.py`: full paired training, scheduling, gradient accumulation/clipping,
  model selection and test reporting.
- `metrics.py`: validation threshold tuning and fraud metrics.
- `plotting.py`: loss figures.
- `tests/test_architecture.py`: minimal architecture and preprocessing tests.

To run the tests: `python -m unittest discover -s tests -v`.

## Research experiments

See [docs/README.md](docs/README.md) for experiment records, figures, results, and planned datasets.

Capacity sweep (Experiment 02; SiLU MLP vs SwiGLU, 4 scales × 3 seeds):

```bash
python -u exp2.py --csv data/creditcard.csv 2>&1 | tee exp2_run.log
```

Outputs are isolated in `exp2/`; historical `results/` is unaffected. Run `python -m unittest discover -s tests -v` before training.

## Experiment 03 — Multi-dataset transfer

See [docs/04_multi_dataset_probe.md](docs/04_multi_dataset_probe.md). One seed across Forest Cover, Mushrooms and SUSY, four parameter scales, SiLU vs SwiGLU. Data download: `bash scripts/download_exp3.sh`; training: `python -u exp3.py`. Compact outputs: `results/exp03/{config.json,datasets.csv,history.csv,metrics.csv}`.

Exp 03 loss curves: `results/exp03/{forest,mushrooms,susy}_loss_curves.png` (4 × 2 panels per dataset). Regenerate without training: `python exp3_plots.py --results-dir results/exp03`.
