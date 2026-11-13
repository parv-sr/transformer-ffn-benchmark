# Transformer FFN Benchmark

An evolving research project on **FT-style Transformers for tabular classification** and **gated feed-forward networks (FFNs)**.

Rather than optimizing for a single benchmark, we investigate *when and why* attention, model capacity and FFN gating improve generalization.

## Research directions

- **FFN design:** Compare parameter-matched SiLU/GELU MLPs with SwiGLU; study ranking quality, calibration and optimization.
- **Capacity and overfitting:** Examine model scaling, generalization gaps and data efficiency.
- **Data-dependent performance:** Identify which feature structures, interactions, label noise and class imbalance predict architectural benefit.
- **Classical ML baselines:** Compare Transformer variants against competitive gradient-boosted trees.

## Architecture

Encoder-only, bidirectional, pre-LayerNorm Transformer blocks with feature-specific numerical projections, learned categorical embeddings, CLS classification and a configurable FFN. The implementation is **FT-style**, not an assertion of exact correspondence to any specific published FT-Transformer.

## Experiments

| Experiment | Aim | Status |
|---|---|---|
| [01](experiments/exp01/README.md) | Ungated vs gated FFN | Completed |
| [02](experiments/exp02/README.md) | Transformer capacity scaling | Completed (large run interrupted) |
| [03](experiments/exp03/README.md) | Cross-dataset probe | Completed |
| [04](experiments/exp04/README.md) | Six-seed SiLU/SwiGLU vs XGBoost | Ready to run |
| [05](experiments/exp05/README.md) | Controlled study of dataset properties | Planned |

See [experiment conventions](experiments/README.md) and [research log](docs/README.md). Each experiment's README contains dataset preparation, protocol, commands and interpretation limits.

## Quick start

From repository root, using [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -r requirements.txt
uv run python -m unittest discover -s experiments -t . -p 'test_*.py' -v
```

Run individual experiments as Python modules; for example:

```bash
uv run python -u -m experiments.exp04.exp04 2>&1 | tee exp04_run.log
```

Data downloads and output conventions are documented by experiment. Results are kept out of Git, typically under `results/expNN/`, with compact JSON configurations, CSV metrics/histories and loss-curve PNGs.

```text
experiments/  # exp01, exp02, exp03, exp04, exp05, ...
models/       # Shared model architectures, FFNs and tokenizers
metrics.py    # Shared evaluation
docs/         # Research index and datasets roadmap
results/      # Untracked generated outputs
```

This is ongoing empirical research. Single-dataset findings and small-seed experiments should not be interpreted as universal architectural conclusions.

---

**Authored by Parv Sharma, FLAME University.**
