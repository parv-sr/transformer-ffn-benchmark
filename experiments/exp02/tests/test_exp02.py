"""Experiment 02 architecture and tokenizer smoke tests."""
import unittest
import torch
from experiments.exp02.exp02 import CAPACITIES, count_parameters
from models.mixed_feature_tokenizer import MixedFeatureTokenizer
from models import GELUTransformer, SwiGLUTransformer


class Experiment2Tests(unittest.TestCase):
    def test_capacity_counts_and_no_dropout(self):
        previous = 0
        for name, spec in CAPACITIES.items():
            d, depth, heads = spec["d_model"], spec["depth"], spec["heads"]
            a = GELUTransformer(d_model=d, depth=depth, num_heads=heads,
                                ffn_hidden_dim=4*d, mlp_activation="silu")
            b = SwiGLUTransformer(d_model=d, depth=depth, num_heads=heads,
                                  ffn_hidden_dim=round(8*d/3))
            p, q = count_parameters(a), count_parameters(b)
            self.assertGreater(p, previous, name)
            self.assertLess(abs(p-q)/p, 0.011)
            self.assertFalse(any(isinstance(layer, torch.nn.Dropout) for layer in a.modules()))
            self.assertEqual(tuple(a(torch.randn(2, 30)).shape), (2, 2))
            previous = p

    def test_mixed_categorical_embedding(self):
        tok = MixedFeatureTokenizer(2, [3, 7], 16)
        x = torch.randn(5, 2)
        cats = torch.tensor([[1,2], [3,7], [0,0], [2,4], [1,7]], dtype=torch.long)
        out = tok(x, cats)
        self.assertEqual(tuple(out.shape), (5, 5, 16))
        out.sum().backward()
        self.assertIsNotNone(tok.embedding.weight.grad)
        self.assertFalse(torch.allclose(out[0, 3], out[0, 4]))
        with self.assertRaises(ValueError):
            tok(x, torch.full((5, 2), 99, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
