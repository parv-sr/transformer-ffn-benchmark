"""Dataset definitions and one shared, leakage-safe preprocessing/split protocol."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"


class FraudDataset(Dataset):
    """Features: FloatTensor[N, 30]; targets: LongTensor[N]."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.int64))
        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same length")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@dataclass
class FraudSplits:
    train: FraudDataset
    val: FraudDataset
    test: FraudDataset
    scaler: StandardScaler
    num_rows_original: int
    num_rows_after_dedup: int


def load_fraud_splits(csv_path: str | Path, seed: int, drop_duplicates: bool = True) -> FraudSplits:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {path}. Download mlg-ulb/creditcardfraud and place "
            "creditcard.csv there (see README.md)."
        )

    df = pd.read_csv(path)
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    original_rows = len(df)
    # Identical feature vectors in different splits would leak information.
    # Removing duplicate features (not merely duplicate full rows) prevents this.
    if drop_duplicates:
        df = df.drop_duplicates(subset=FEATURE_COLUMNS, keep="first").reset_index(drop=True)

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if not np.isfinite(x).all():
        raise ValueError("Dataset has missing/infinite feature values")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("The Class column must contain binary labels 0 and 1")

    # 70 / 15 / 15, with fraud proportions preserved in all partitions.
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    # IMPORTANT: fit only on the training partition. Reuse for both models.
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return FraudSplits(
        train=FraudDataset(x_train, y_train),
        val=FraudDataset(x_val, y_val),
        test=FraudDataset(x_test, y_test),
        scaler=scaler,
        num_rows_original=original_rows,
        num_rows_after_dedup=len(df),
    )
