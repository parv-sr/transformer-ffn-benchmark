import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.exp03.exp03_data import load_dataset, sample_rows
from experiments.exp03.exp03_model import MixedBinaryTransformer
from experiments.exp01.train import num_parameters, synchronize_shared_weights


class Exp03Tests(unittest.TestCase):
    def _fixtures(self, root):
        rng = np.random.default_rng(321)
        base = Path(root)
        (base/"forest").mkdir()
        (base/"mushrooms").mkdir()
        (base/"susy").mkdir()
        n = 240
        forest = pd.DataFrame(rng.normal(size=(n, 10)),
                              columns=[f"numeric_{i}" for i in range(10)])
        for i in range(1, 5):
            forest[f"Wilderness_Area{i}"] = (rng.integers(1,5,size=n)==i).astype(int)
        # Guarantee exactly one wilderness flag.
        w = rng.integers(0,4,size=n)
        for i in range(1,5):
            forest[f"Wilderness_Area{i}"] = (w==i-1).astype(int)
        s = rng.integers(0,40,size=n)
        for i in range(1,41):
            forest[f"Soil_Type{i}"] = (s==i-1).astype(int)
        forest["Cover_Type"] = np.tile([1,2,3,1],n//4)
        forest.to_csv(base/"forest"/"covtype.csv",index=False)

        mushrooms = pd.DataFrame({
            "id": np.arange(n), "class": np.tile(["e","p"],n//2),
            "cap-diameter": rng.normal(size=n), "stem-height": rng.normal(size=n),
            "stem-width": rng.normal(size=n),
            "cap-shape": np.tile(["x","f",None], n//3),
            "odor": np.tile(["a","n"], n//2)
        })
        mushrooms.to_csv(base/"mushrooms"/"train.csv",index=False)
        susy = np.column_stack((np.tile([0,1], n//2), rng.normal(size=(n,18))))
        pd.DataFrame(susy).to_csv(base/"susy"/"SUSY.csv",index=False,header=False)

    def test_all_dataset_schemas(self):
        with tempfile.TemporaryDirectory() as d:
            self._fixtures(d)
            for name in ("forest","mushrooms","susy"):
                with self.subTest(dataset=name):
                    data = load_dataset(Path(d), name, max_rows=120, seed=42)
                    self.assertEqual(data.sampled_rows, 120)
                    self.assertEqual(len(data.train)+len(data.val)+len(data.test), 120)
                    self.assertGreater(data.population_rows, 120)
                    ncat=len(data.cardinalities)
                    self.assertEqual(data.train.tensors[1].shape[1], ncat)
                    self.assertTrue(torch.isfinite(data.train.tensors[0]).all())
                    args=dict(num_numeric=len(data.numeric_columns),
                              cat_cardinalities=data.cardinalities,
                              d_model=16, depth=2, num_heads=2)
                    model_a=MixedBinaryTransformer(**args,kind="silu")
                    model_b=MixedBinaryTransformer(**args,kind="swiglu")
                    self.assertLess(abs(num_parameters(model_a)-num_parameters(model_b))
                                    /num_parameters(model_a), 0.02)
                    self.assertGreater(synchronize_shared_weights(model_a,model_b),0)
                    x, c, y = data.train.tensors
                    logits=model_a(x[:4],c[:4])
                    self.assertEqual(logits.shape,(4,2))
                    torch.nn.functional.cross_entropy(logits,y[:4]).backward()
                    self.assertIsNotNone(model_a.head.weight.grad)

    def test_susy_sampling_repeatable(self):
        with tempfile.TemporaryDirectory() as d:
            self._fixtures(d)
            p=Path(d)/"susy"/"SUSY.csv"
            a,n1=sample_rows(p,"susy",110,42)
            b,n2=sample_rows(p,"susy",110,42)
            self.assertEqual((n1,n2),(240,240))
            pd.testing.assert_frame_equal(a,b)


if __name__ == "__main__":
    unittest.main()
