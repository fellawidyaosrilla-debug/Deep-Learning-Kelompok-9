"""Tes unit yang tidak membutuhkan unduhan dataset asli."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.preprocessing import read_antenna
from src.dataset import group_split, normalize_splits, coords_to_zone
from src.models import MLP, CNNLSTM


class ProjectTests(unittest.TestCase):
    def test_model_shapes(self):
        x = torch.randn(2, 6, 15, 30)
        self.assertEqual(tuple(MLP()(x).shape), (2, 2))
        self.assertEqual(tuple(CNNLSTM()(x).shape), (2, 2))

    def test_preprocessing_interpolates_invalid(self):
        t = np.arange(25, dtype=float)
        data = {f'amplitude_{k}': 5 + np.sin(t * 1.2 + k) for k in range(30)}
        data.update({f'phase_{k}': 0.1 * t + 0.5 * k for k in range(30)})
        data['amplitude_9'][10] = -np.inf
        with tempfile.TemporaryDirectory() as temp:
            filepath = Path(temp) / 'antenna_1_0_0.csv'
            pd.DataFrame(data).to_csv(filepath, index=False)
            amp, ph = read_antenna(filepath, butterworth=True)
            amp_raw, _ = read_antenna(filepath, butterworth=False)
        self.assertEqual(amp.shape, (25, 30))
        self.assertEqual(ph.shape, (25, 30))
        self.assertTrue(np.isfinite(amp).all())
        self.assertTrue(np.isfinite(ph).all())
        self.assertGreater(float(np.abs(amp - amp_raw).mean()), 0.001)

    def test_group_stratification_and_no_leakage(self):
        y = np.array([[x, yy] for x in range(9) for yy in range(10)], dtype=np.float32)
        groups = np.arange(len(y))
        tr, va, te = group_split(y, groups)
        self.assertEqual(len(tr) + len(va) + len(te), len(y))
        self.assertFalse(set(tr) & set(va))
        self.assertFalse(set(tr) & set(te))
        self.assertFalse(set(va) & set(te))
        for selected in (tr, va, te):
            self.assertEqual(len(set(coords_to_zone(y[selected]))), 4)

    def test_training_only_normalization(self):
        tr = np.ones((2, 6, 15, 30), dtype=np.float32) * 4
        va = np.ones((1, 6, 15, 30), dtype=np.float32) * 8
        te = np.ones((1, 6, 15, 30), dtype=np.float32) * 40
        (a, b, c), mu, std = normalize_splits(tr, va, te)
        self.assertTrue(np.allclose(a, 0))
        self.assertTrue(np.allclose(b, 4))
        self.assertTrue(np.allclose(c, 36))
        self.assertTrue(np.allclose(mu, 4))
        self.assertTrue(np.allclose(std, 1))


if __name__ == '__main__':
    unittest.main()
