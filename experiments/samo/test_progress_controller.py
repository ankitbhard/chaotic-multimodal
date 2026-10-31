"""Regression tests for RA-SAMO v1 training-only progress controller."""
import math
import unittest
from experiments.samo.progress_controller import ProgressController


class TestProgressController(unittest.TestCase):
    def test_neutral_and_equal_progress(self):
        c = ProgressController(ema=0, warmup=0)
        self.assertEqual(c.update(0, 2, 4), (1.0, 1.0))
        self.assertEqual(c.update(1, 1, 2), (1.0, 1.0))

    def test_lagging_image_gets_larger_multiplier(self):
        c = ProgressController(ema=0, gain=2, warmup=0)
        c.update(0, 2, 2)
        image, gene = c.update(1, 1.9, 1.0)
        self.assertGreater(image, 1)
        self.assertLess(gene, 1)
        self.assertLessEqual(image, 1.5)  # product not preserved after clamping

    def test_bounds_and_nonfinite(self):
        c = ProgressController(ema=0, gain=100, warmup=0)
        c.update(0, 2, 2)
        self.assertEqual(c.update(1, 2, 1), (1.5, 0.5))
        self.assertEqual(c.update(2, math.nan, 1), (1.5, 0.5))

    def test_warmup_and_invalid_config(self):
        c = ProgressController(ema=0, warmup=2)
        self.assertEqual(c.update(0, 2, 2), (1, 1))
        self.assertEqual(c.update(1, 1.9, 1), (1, 1))
        self.assertGreater(c.update(2, 1.8, 0.8)[0], 1)
        with self.assertRaises(ValueError):
            ProgressController(ema=1)


if __name__ == "__main__":
    unittest.main()
