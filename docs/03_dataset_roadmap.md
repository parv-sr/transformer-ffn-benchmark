# Exp 03 — Additional datasets (planned)

**Goal:** Test FFN generalization across domains, with hundreds of thousands of rows, mostly balanced labels, no PCA features and mixed feature types.

| Order | Dataset | Labeled rows | Features | Classes / balance | Preparation |
|---|---|---:|---|---|---|
| 1 | [Forest Cover Type](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset) | 581,012 overall; 495,141 in classes 1/2 | 10 numeric + two original one-hot groups | Binary subset: 42.8% / 57.2% | Select types 1/2, convert 4 wilderness and 40 soil flags into two categorical IDs |
| 2 | [Poisonous Mushrooms (Kaggle S4E8)](https://www.kaggle.com/competitions/playground-series-s4e8/data) | 3,116,945 train | 3 numerical + 18 categorical | Binary, approx. 55% / 45% | Synthetic, many missing/noisy categories; competition rules may require acceptance |
| 3 | [SUSY](https://www.kaggle.com/datasets/janus137/supersymmetry-dataset) | 5,000,000 | 18 numerical | Binary; verify local prevalence | No PCA; fixed 300K–500K subsample for compute parity |

Forest's original **seven-class** task is imbalanced. Only the *defined binary types 1/2 subset* matches our balance criterion. Verify actual class counts after download for every dataset. Do not rebalance held-out samples.

## Mixed feature tokenizer

Prototype: [mixed_feature_tokenizer.py](../models/mixed_feature_tokenizer.py). This is **not used by Exp 02**.

| Input | Token rule | Learned parameters |
|---|---|---|
| Numeric column j | x_j × W_j + b_j | Two d-vectors per feature |
| Categorical column j | E_j[value_id] | (K_j+1) × d lookup; 0 = missing/unseen |
| Classification | Prepend learned [CLS] | One d-vector |

Each categorical field has its own disjoint embedding rows; total rows = sum_j(K_j+1). The target label must never be an input feature.

Before running Exp 03: fit vocabularies/scalers on training data only, handle missing and unknown values, support multiple classes in the classifier head, and compare on identical data splits. Report accuracy, AUROC, AP, F1, cross-entropy and calibration, plus compute and parameter counts.
