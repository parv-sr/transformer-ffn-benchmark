"""Exp 04 integration smoke tests; small synthetic Forest CSV, CPU XGBoost."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.exp03.exp03_data import load_dataset, sample_rows
from experiments.exp04.exp04 import fit_xgb, paired_summary, xgb_inputs
from experiments.exp04.exp04_plots import regenerate_plots


def make_forest(root: Path):
    rng = np.random.default_rng(18)
    n = 320
    df = pd.DataFrame(rng.normal(size=(n, 10)),
                      columns=[f"numeric_{i}" for i in range(10)])
    wild = rng.integers(0, 4, size=n)
    soil = rng.integers(0, 40, size=n)
    for i in range(4):
        df[f"Wilderness_Area{i+1}"] = (wild == i).astype(int)
    for i in range(40):
        df[f"Soil_Type{i+1}"] = (soil == i).astype(int)
    df["Cover_Type"] = np.where(df["numeric_0"] + 0.3*df["numeric_1"] > 0, 2, 1)
    folder = root/"forest"
    folder.mkdir(parents=True)
    csv = folder/"covtype.csv"
    df.to_csv(csv, index=False)
    return csv


class Experiment04Tests(unittest.TestCase):
    def test_fixed_sample_and_split_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv = make_forest(root)
            sample_a, _ = sample_rows(csv, "forest", 200, 42)
            sample_b, _ = sample_rows(csv, "forest", 200, 42)
            pd.testing.assert_frame_equal(sample_a, sample_b)
            a = load_dataset(root, "forest", 200, seed=42, sample_seed=42)
            b = load_dataset(root, "forest", 200, seed=43, sample_seed=42)
            self.assertEqual(a.sampled_rows, b.sampled_rows)
            self.assertEqual((len(a.train),len(a.val),len(a.test)), (140,30,30))
            # Train-only category vocabularies may differ across split seeds.
            self.assertFalse(torch.equal(a.train.tensors[0], b.train.tensors[0]))
            x_train, x_val, x_test = xgb_inputs(a)
            self.assertEqual(len(x_train[0]), len(a.train))
            self.assertEqual(str(x_train[0]["soil"].dtype), "category")
            self.assertEqual(list(x_train[0]["soil"].cat.categories),
                             list(x_val[0]["soil"].cat.categories))

    def test_xgboost_and_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_forest(root)
            data = load_dataset(root, "forest", 220, seed=42, sample_seed=42)
            args = SimpleNamespace(xgb_estimators=20, xgb_device="cpu", xgb_jobs=1)
            history, metrics = fit_xgb(data, args, seed=42)
            self.assertGreater(len(history), 0)
            self.assertTrue(0 <= metrics["test_ap"] <= 1)
            self.assertTrue(0 <= metrics["test_f1"] <= 1)
            self.assertIn("test_brier", metrics)
            df = pd.DataFrame(history)
            df.to_csv(root/"history.csv", index=False)
            path = regenerate_plots(root/"history.csv", root)
            self.assertIsNotNone(path)
            self.assertGreater(path.stat().st_size, 1000)
            paired_summary([metrics], root/"summary.csv")
            self.assertTrue((root/"summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
