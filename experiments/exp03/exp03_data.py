"""Experiment 03: bounded-memory sampling and leakage-safe mixed tabular features."""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

DATASETS = ("forest", "mushrooms", "susy")
NAMES = {
    "forest": ("covtype.csv", "forest_cover_type.csv"),
    "mushrooms": ("train.csv",),
    "susy": ("SUSY.csv", "SUSY.csv.gz", "susy.csv", "susy.csv.gz"),
}
MUSHROOM_NUMERIC = ("cap-diameter", "stem-height", "stem-width")


@dataclass
class PreparedData:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    cardinalities: tuple[int, ...]
    numeric_columns: tuple[str, ...]
    category_columns: tuple[str, ...]
    sampled_rows: int
    population_rows: int
    positive_rate: float


def find_source(root: Path, dataset: str) -> Path:
    folder = root / dataset
    for name in NAMES[dataset]:
        path = folder / name
        if path.is_file():
            return path
    matches = sorted([*folder.glob("*.csv"), *folder.glob("*.csv.gz")])
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot uniquely locate {dataset} CSV in {folder}. "
        f"Expected one of {NAMES[dataset]}; found {[x.name for x in matches]}"
    )


def _susy_csv_options(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first = handle.readline().split(",")[0].strip()
    try:
        float(first)
        return {"header": None, "names": ["target", *[f"f{i:02d}" for i in range(1, 19)]]}
    except ValueError:
        return {"header": 0}


def _forest_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    wilderness = sorted((c for c in frame if re.fullmatch(r"Wilderness_Area_?\d+", str(c), flags=re.I)),
                        key=lambda c: int(re.search(r"\d+$", c).group()))
    soils = sorted((c for c in frame if re.fullmatch(r"Soil_Type_?\d+", str(c), flags=re.I)),
                   key=lambda c: int(re.search(r"\d+$", c).group()))
    if len(wilderness) != 4 or len(soils) != 40:
        raise ValueError("Forest CSV must contain 4 Wilderness_Area and 40 Soil_Type indicator columns")
    excluded = set([*wilderness, *soils, "Cover_Type", "Id", "id"])
    numeric = [c for c in frame if c not in excluded]
    if len(numeric) != 10:
        raise ValueError(f"Forest CSV expected 10 numerical columns, got {numeric}")
    return numeric, wilderness, soils


def _make_forest(frame: pd.DataFrame) -> pd.DataFrame:
    if "Cover_Type" not in frame:
        raise ValueError("Forest input must have Cover_Type (full labeled UCI dataset, not competition test)")
    frame = frame[frame["Cover_Type"].isin([1, 2])].copy()
    if len(frame) == 0:
        return frame
    numeric, wild, soils = _forest_columns(frame)
    wild_values = frame[wild].to_numpy()
    soil_values = frame[soils].to_numpy()
    if not ((wild_values.sum(axis=1) == 1).all() and (soil_values.sum(axis=1) == 1).all()):
        raise ValueError("Expected valid one-hot forest wilderness/soil features")
    result = frame[numeric].copy()
    result["wilderness"] = np.argmax(wild_values, axis=1).astype(str)
    result["soil"] = np.argmax(soil_values, axis=1).astype(str)
    result["target"] = (frame["Cover_Type"] == 2).astype(int)
    return result


def _make_mushrooms(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {"class", *MUSHROOM_NUMERIC}
    if not expected.issubset(frame):
        raise ValueError(f"Mushroom train.csv missing columns: {expected - set(frame)}")
    result = frame.drop(columns=["id", "Id"], errors="ignore").rename(columns={"class": "target"})
    result["target"] = result["target"].map({"e": 0, "p": 1})
    return result


def _make_susy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] != 19:
        raise ValueError("SUSY CSV must contain one label and 18 numerical features")
    result = frame.copy()
    result.columns = ["target", *[f"f{i:02d}" for i in range(1, 19)]]
    return result


TRANSFORMS = {"forest": _make_forest, "mushrooms": _make_mushrooms, "susy": _make_susy}


def sample_rows(path: Path, dataset: str, max_rows: int, seed: int) -> tuple[pd.DataFrame, int]:
    """Uniform reproducible min-priority sample from a CSV stream.

    Holds at most max_rows + chunksize rows in memory; samples before splitting.
    """
    options = _susy_csv_options(path) if dataset == "susy" else {}
    rng = np.random.default_rng(seed)
    selected, priorities = None, np.empty(0, dtype=np.float64)
    total = 0
    for frame in pd.read_csv(path, chunksize=75_000, low_memory=False, **options):
        frame = TRANSFORMS[dataset](frame)
        total += len(frame)
        if frame.empty:
            continue
        score = rng.random(len(frame))
        if selected is None:
            selected = frame
            priorities = score
        else:
            selected = pd.concat((selected, frame), ignore_index=True)
            priorities = np.concatenate((priorities, score))
        if len(selected) > max_rows:
            keep = np.sort(np.argpartition(priorities, max_rows - 1)[:max_rows])
            selected = selected.iloc[keep].reset_index(drop=True)
            priorities = priorities[keep]
    if selected is None or len(selected) < 100:
        raise ValueError(f"Insufficient labeled rows in {path}")
    # Stable row order, independent of chunk boundaries.
    selected = selected.iloc[np.argsort(priorities)].reset_index(drop=True)
    return selected, total


def _prepare_split(frame: pd.DataFrame, numeric: list[str], categorical: list[str],
                   vocabularies: dict, scaler: StandardScaler, medians: pd.Series) -> TensorDataset:
    if numeric:
        values = frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(medians).fillna(0)
        x_num = scaler.transform(values.to_numpy(dtype=np.float64)).astype(np.float32)
    else:
        x_num = np.empty((len(frame), 0), dtype=np.float32)
    x_cat = np.zeros((len(frame), len(categorical)), dtype=np.int64)
    for j, col in enumerate(categorical):
        x_cat[:, j] = (frame[col].astype("string").map(vocabularies[col])
                       .fillna(0).to_numpy(dtype=np.int64))
    y = frame["target"].to_numpy(dtype=np.int64)
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x_num)),
                         torch.from_numpy(x_cat), torch.from_numpy(y))


def load_dataset(root: Path, dataset: str, max_rows: int, seed: int) -> PreparedData:
    path = find_source(root, dataset)
    frame, population = sample_rows(path, dataset, max_rows=max_rows, seed=seed)
    if frame["target"].isna().any():
        raise ValueError(f"{dataset}: unknown class label(s)")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError(f"{dataset}: expected two class labels 0 and 1")
    if dataset == "mushrooms":
        numeric = list(MUSHROOM_NUMERIC)
    elif dataset == "forest":
        numeric = [c for c in frame if c not in ("target", "wilderness", "soil")]
    else:
        numeric = [c for c in frame if c != "target"]
    categorical = [c for c in frame if c not in [*numeric, "target"]]
    indices = np.arange(len(frame))
    train, temp = train_test_split(indices, test_size=0.3, stratify=frame["target"],
                                   random_state=seed)
    val, test = train_test_split(temp, test_size=0.5,
                                stratify=frame.iloc[temp]["target"], random_state=seed)
    subsets = [frame.iloc[i].copy() for i in (train, val, test)]
    train_frame = subsets[0]
    vocabs = {}
    for col in categorical:
        unique = train_frame[col].astype("string").dropna().unique().tolist()
        vocabs[col] = {value: i + 1 for i, value in enumerate(sorted(unique))}
    if numeric:
        n_train = train_frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians = n_train.median().fillna(0)
        scaler = StandardScaler().fit(n_train.fillna(medians).fillna(0))
    else:
        medians = pd.Series(dtype="float64")
        scaler = StandardScaler()
    prepared = [_prepare_split(s, numeric, categorical, vocabs, scaler, medians) for s in subsets]
    return PreparedData(prepared[0], prepared[1], prepared[2],
                        tuple(len(v) for v in vocabs.values()), tuple(numeric),
                        tuple(categorical), len(frame), population,
                        float(frame["target"].mean()))
