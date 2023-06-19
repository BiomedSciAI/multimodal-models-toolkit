"""Unit test for models defined in mmmt.models.classic"""

import unittest
import numpy as np
from mmmt.models.classic.late_fusion import LateFusion
from mmmt.models.classic.uncertainty_late_fusion import UncertaintyLateFusion


class LateFusionTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_late_fusion(self):
        n_mods = 20
        n_samples = 1000
        n_classes = 10

        lf = LateFusion(n_mods, n_classes)

        predictions_test = np.random.rand(n_mods, n_samples, n_classes)
        fused_test = lf.apply_fusion(predictions_test)
        self.assertEqual(fused_test.shape[0], predictions_test.shape[1])


class UncertaintyFusionTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_uncertainty_late_fusion(self):
        n_mods = 20
        n_samples = 1000
        n_classes = 10

        predictions_valid = np.random.rand(n_mods, n_samples, n_classes)
        GT_valid = np.random.rand(n_samples, n_classes)

        ulf = UncertaintyLateFusion(n_mods, n_classes)
        ulf.k = 5

        ulf.compute_fusion_weights(predictions_valid, GT_valid)

        predictions_test = np.random.rand(n_mods, n_samples, n_classes)
        fused_test = ulf.apply_fusion(predictions_test)
        self.assertEqual(fused_test.shape[0], predictions_test.shape[1])
