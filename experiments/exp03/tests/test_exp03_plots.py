"""Exp 03 plots are reproducible from CSV alone."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.exp03.exp03_plots import regenerate_plots


class LossPlotTests(unittest.TestCase):
    def test_one_png_per_available_dataset_and_partial_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history = pd.DataFrame([
                dict(dataset=dataset, capacity="nano", model=model,
                     epoch=epoch, train_ce=0.5 / epoch, val_ce=0.3 + 0.01*epoch)
                for dataset in ("forest", "susy")
                for model in ("silu", "swiglu")
                for epoch in (1, 2, 3)
            ])
            history.to_csv(root / "history.csv", index=False)
            outputs = regenerate_plots(root / "history.csv", root)
            self.assertEqual({path.name for path in outputs},
                             {"forest_loss_curves.png", "susy_loss_curves.png"})
            self.assertTrue(all(path.stat().st_size > 1000 for path in outputs))
            # Idempotent and safe to regenerate after resumed training.
            self.assertEqual({p.name for p in regenerate_plots(root / "history.csv", root)},
                             {p.name for p in outputs})
            self.assertEqual(len(list(root.glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
