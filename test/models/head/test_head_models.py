"""Unit test for models defined in mmmt.models.head"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
import unittest
import torch
from mmmt.models.head.mlp import MLP


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        # preparing dummy data
        self.num_nodes = 10
        self.head_in_size = 32
        self.out_size = 5
        self.batch_size = 2

        self.h = torch.rand(self.num_nodes, self.batch_size, self.head_in_size)
        pass

    def test_mlp(self):
        # preparing head model object
        head_hidden_size = [20, 10]
        dropout = 0.5

        MLP_obj = MLP(
            self.num_nodes, self.head_in_size, head_hidden_size, self.out_size, dropout
        )

        h = MLP_obj(self.h)

        self.assertEqual(self.out_size * self.batch_size, torch.numel(h))

    def tearDown(self):
        """Tear down the tests."""
        pass
