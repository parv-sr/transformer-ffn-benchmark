# Experiments

Every experiment gets one folder `experiments/expNN/`.

| Experiment | Runner | Scope |
|---|---|---|
| [Exp 01](exp01/README.md) | `python -m experiments.exp01.train` | Credit-card FFN comparison |
| [Exp 02](exp02/README.md) | `python -m experiments.exp02.exp02` | Capacity sweep |
| [Exp 03](exp03/README.md) | `python -m experiments.exp03.exp03` | Multi-dataset probe |

**Future experiments:** use `experiments/exp04/`, `exp05/`, etc. Keep each runner, preprocessing, plotting, tests, and one short Markdown protocol within its own folder. Shared model primitives stay under `models/`; shared metrics stay in `metrics.py`. Keep generated results in ignored output directories, not in Git.

Run all commands **from the repository root**:

```bash
python -m unittest discover -s experiments -t . -p 'test_*.py' -v
python -m experiments.exp03.exp03_plots --results-dir results/exp03
```

Exp 03 figures appear as `results/exp03/{forest,mushrooms,susy}_loss_curves.png`. Existing histories can be plotted without retraining.
