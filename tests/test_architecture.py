import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataloaders import build_dataloaders
from dataset import FEATURE_COLUMNS, load_fraud_splits
from metrics import compute_fraud_metrics, validation_f1_threshold
from models import GELUTransformer, SwiGLUTransformer
from models.ffn import GELUFFN, SwiGLUFFN
from train import num_parameters, synchronize_shared_weights


class ArchitectureTests(unittest.TestCase):
    def test_token_shapes_gradients_and_no_dropout(self):
        for cls, width in [(GELUTransformer, 128), (SwiGLUTransformer, 86)]:
            with self.subTest(model=cls.__name__):
                model = cls(d_model=32, depth=2, num_heads=4, ffn_hidden_dim=width)
                self.assertFalse(any(isinstance(layer, torch.nn.Dropout) for layer in model.modules()))
                self.assertTrue(all(getattr(layer, "dropout", 0) == 0 for layer in model.modules()
                                    if isinstance(layer, torch.nn.MultiheadAttention)))
                x = torch.randn(3, 30)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (3, 2))
                probs = model.predict_proba(x)
                torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3))
                logits.sum().backward()
                self.assertIsNotNone(model.tokenizer.weight.grad)

    def test_full_feature_interaction(self):
        torch.manual_seed(13)
        model = GELUTransformer(d_model=32, depth=2, num_heads=4, ffn_hidden_dim=128).eval()
        x = torch.randn(2, 30)
        x_changed = x.clone()
        x_changed[:, -1] += 10
        # No causal mask: a change to any feature token can affect CLS logits.
        self.assertFalse(torch.allclose(model(x), model(x_changed)))

    def test_matched_parameters_and_common_init(self):
        gelu = GELUTransformer(d_model=96, depth=4, num_heads=4, ffn_hidden_dim=384)
        swiglu = SwiGLUTransformer(d_model=96, depth=4, num_heads=4, ffn_hidden_dim=256)
        self.assertLess(abs(num_parameters(gelu)-num_parameters(swiglu)) / num_parameters(gelu), .005)
        self.assertGreater(synchronize_shared_weights(gelu, swiglu), 0)
        torch.testing.assert_close(gelu.tokenizer.weight, swiglu.tokenizer.weight)

    def test_ffn_outputs(self):
        x = torch.randn(4, 6, 32)
        self.assertEqual(GELUFFN(32, 128)(x).shape, x.shape)
        self.assertEqual(SwiGLUFFN(32, 86)(x).shape, x.shape)

    def test_splits_loader_and_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            n = 160
            rng = np.random.default_rng(9)
            df = pd.DataFrame(rng.normal(size=(n, 30)), columns=FEATURE_COLUMNS)
            df["Class"] = [0] * 140 + [1] * 20
            df = pd.concat([df, df.iloc[[1]]], ignore_index=True)  # duplicate
            path = Path(d) / "creditcard.csv"
            df.to_csv(path, index=False)
            splits = load_fraud_splits(path, seed=42)
            self.assertEqual(splits.num_rows_original, n + 1)
            self.assertEqual(splits.num_rows_after_dedup, n)
            self.assertEqual((len(splits.train), len(splits.val), len(splits.test)), (112, 24, 24))
            self.assertTrue(np.allclose(splits.train.features.numpy().mean(axis=0), 0, atol=1e-5))
            loaders = build_dataloaders(splits, batch_size=16, seed=42)
            x, y = next(iter(loaders.train))
            self.assertEqual(x.shape, (16, 30))
            self.assertEqual(y.dtype, torch.int64)
            true = np.array([0, 0, 1, 1])
            probs = np.array([0.1, 0.3, 0.8, 0.9])
            th = validation_f1_threshold(true, probs)
            metrics = compute_fraud_metrics(true, probs, th)
            self.assertAlmostEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
