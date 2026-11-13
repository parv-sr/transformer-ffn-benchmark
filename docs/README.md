# Research log

| Experiment | Scope | Status | Files |
|---|---|---|---|
| [Exp 01](../experiments/exp01/README.md) | Fraud FFN comparison (GELU, SiLU, SwiGLU) | Results recorded | `experiments/exp01/` |
| [Exp 02](../experiments/exp02/README.md) | Capacity sweep, four scales × three seeds | Results collected | `experiments/exp02/` |
| [Exp 03](../experiments/exp03/README.md) | Three datasets × four scales × two FFNs | Training / analysis | `experiments/exp03/` |
| [Roadmap](03_dataset_roadmap.md) | Next dataset comparisons | Planned | `docs/` |

All future experiment-specific runners, tests, plots and protocols belong under `experiments/expNN/`. Shared model and metrics libraries remain at repository root. Run modules from the repository root with `python -m experiments.expNN.<module>`.
