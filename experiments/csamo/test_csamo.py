import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from experiments.csamo.run import WEIGHTS, CANDIDATES, validation_loaders, choose_mode

class ControllerTests(unittest.TestCase):
    def test_candidates(self):
        self.assertEqual(set(CANDIDATES), set(WEIGHTS))
        self.assertEqual(WEIGHTS["rank"][0], 0)
        self.assertEqual(WEIGHTS["aux"][1], 0)
    def test_validation_split_is_disjoint(self):
        loader = DataLoader(TensorDataset(torch.arange(30)), batch_size=8)
        a, b = validation_loaders(loader, 0.5, 42)
        self.assertFalse(set(a.dataset.indices) & set(b.dataset.indices))
        self.assertEqual(len(a.dataset) + len(b.dataset), 30)
    def test_split_deterministic(self):
        loader = DataLoader(TensorDataset(torch.arange(30)), batch_size=8)
        a, _ = validation_loaders(loader, 0.5, 42)
        b, _ = validation_loaders(loader, 0.5, 42)
        self.assertEqual(a.dataset.indices, b.dataset.indices)
    def test_invalid_small_validation(self):
        loader = DataLoader(TensorDataset(torch.arange(10)), batch_size=4)
        with self.assertRaises(ValueError):
            validation_loaders(loader, 0.5, 42)

if __name__ == "__main__":
    unittest.main()
