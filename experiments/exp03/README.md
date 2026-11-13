# Exp 03 — Single-seed multi-dataset probe

**Question:** Does SwiGLU improve ranking quality across datasets and capacities, beyond ULB fraud?

| Dataset | Source | Input | Label |
|---|---|---|---|
| [Forest Cover Type](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset) | Full 581K-row labeled dataset | 10 numeric, 2 categorical (decoded wilderness/soil) | Binary types 1 vs 2 |
| [Mushrooms (S4E8)](https://www.kaggle.com/competitions/playground-series-s4e8/data) | Kaggle synthetic competition train set | 3 numeric, 18 categorical | Edible vs poisonous |
| [SUSY](https://www.kaggle.com/datasets/janus137/supersymmetry-dataset) | 5M-row physics CSV | 18 numeric | Signal vs background |

**Training design:** 300,000 uniform sampled labeled rows *per dataset*, stratified 70/15/15, seed 42. 4 capacities (nano, small, baseline, large) × 2 models (SiLU MLP, SwiGLU) × 3 datasets = **24 fits**. 12 epochs max, patience 3, AdamW 3e-4, batch size 512, accumulation 2, no dropout; unweighted cross-entropy for balanced tasks; best validation AP; validation-selected F1 threshold. Sampling precedes splitting, with vocabularies, imputation and scaling fitted only on train. The same sample and split are used by both models at every scale.

## Setup / download

Kaggle credentials and competition rule acceptance required. From repo root:

~~~bash
uv pip install -r requirements.txt
bash experiments/exp03/download.sh
python -u -m experiments.exp03.exp03 2>&1 | tee exp03_run.log
~~~

For a preflight without creating outputs: `python -m experiments.exp03.exp03 --dry-run`.

## Compact outputs (no checkpoints/predictions)

~~~text
results/exp03/
  config.json       # immutable run configuration
  datasets.csv      # source/sample counts, label balance, feature counts
  history.csv       # epoch-level train/val CE, validation AP
  metrics.csv       # one row per dataset × capacity × model
  forest_loss_curves.png
  mushrooms_loss_curves.png
  susy_loss_curves.png
~~~

Each PNG contains an eight-panel grid (4 capacities × 2 FFNs), with train/validation CE and empty panels labeled pending. Plots regenerate automatically as fits finish and when an interrupted job restarts. To create PNGs from an existing history without training:

~~~bash
python -m experiments.exp03.exp03_plots --results-dir results/exp03
~~~

On restart, complete fits are skipped. Each fit is committed atomically to the CSVs; model snapshots are held only in RAM. Changing configuration requires a **new output directory**. No saved checkpoints means individual fits cannot resume mid-training.

## Results (pending)

| Dataset | Nano ΔAP | Small ΔAP | Baseline ΔAP | Large ΔAP |
|---|---:|---:|---:|---:|
| Forest | TBD | TBD | TBD | TBD |
| Mushrooms | TBD | TBD | TBD | TBD |
| SUSY | TBD | TBD | TBD | TBD |

ΔAP = SwiGLU − SiLU; test metrics are measured once, on the validation-selected epoch. The probe is exploratory: one seed and a stratified random split do not establish robustness. Parameter counts vary by dataset because tokenizer dimensions and category vocabularies vary. The 300K sample is a compute-control measure, not a full-dataset benchmark.

Future: paired multi-seed runs, calibration and F1/AP comparisons, fixed-split controls.
