"""Unit test for complex_module.core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import unittest
from mmmt.data.graph.dgl_data_loader import DGLFileLoader
from dgl import DGLGraph
import torch


class DGLFileLoaderTestCase(unittest.TestCase):
    """DGLFileLoaderTestCase     class."""

    def setUp(self):
        """Setting up the test."""
        self.existing_datasets = ["AIFB"]
        self.non_existing_dataset = "this_dataset_does_not_exist"
        pass

    def test_dataset_names(self):
        """Test that supported datasets return the correct types."""
        for dataset_name in self.existing_datasets:
            loader = DGLFileLoader(dataset_name, [0.3, 0.3, 0.4], 0)
            g, labels, data_splits, n_classes, num_rels = loader.build_graph()
            self.assertIsInstance(g, DGLGraph)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertIsInstance(data_splits, list)
            self.assertIsInstance(n_classes, int)
            self.assertIsInstance(num_rels, int)

    def test_incorrect_splits(self):
        """Test that supported datasets return the correct types."""
        with self.assertRaises(AssertionError):
            loader = DGLFileLoader("ACM", [0.6, 0.3, 0.4], 0)

    def test_non_existing_dataset(self):
        """Test that a non supported dataset raises an error"""
        with self.assertRaises(ValueError):
            DGLFileLoader(self.non_existing_dataset, [0.3, 0.3, 0.4], 0)

    def test_not_enough_parameters(self):
        """Test that a non supported dataset raises an error"""
        with self.assertRaises(TypeError):
            DGLFileLoader()

    def tearDown(self):
        """Tear down the tests."""
        pass
