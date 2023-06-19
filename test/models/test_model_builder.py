"""Unit test for mmmt.models.model_builder"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
import unittest
import dgl
import torch
import numpy as np
import mmmt
from mmmt.data.graph.graph_to_graph import GraphTransform


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        # preparing dummy data
        num_nodes = 10
        num_edges = 100
        graph = dgl.rand_graph(num_nodes, num_edges)
        node_emb_dim = 4
        features = np.random.rand(num_nodes, node_emb_dim).astype(np.float32)

        self.graph_model_input = {
            "graph_module": {
                "module_identifier": "rgcn",
                "thresh_q": 0.95,
                "node_emb_dim": node_emb_dim,
                "gl_hidden_size": [2],
                "num_att_heads": 4,
                "num_bases": 8,
                "n_layers": 1,
            },
            "head_module": {
                "head_hidden_size": [100, 20],
                "dropout": 0.5,
            },
        }

        self.out_size = 5

        GT = GraphTransform(
            graph,
            features,
            graph_module=self.graph_model_input["graph_module"]["module_identifier"],
        )
        derived_graph, features_multigraph = GT.transform()
        self.graph_dict = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        self.graph_sample = derived_graph

        self.batch_size = 1

        pass

    def test_model_builder(self):
        """Test ModelBuilder."""
        head_hidden_size = self.graph_model_input["head_module"]["head_hidden_size"]
        dropout = self.graph_model_input["head_module"]["dropout"]
        MC = mmmt.models.graph.module_configurator.ModuleConfigurator(
            self.graph_model_input["graph_module"]
        )
        graph_model, head_in_size, head_num_nodes = MC.get_module(self.graph_sample)
        head_model = mmmt.models.head.mlp.MLP(
            head_num_nodes, head_in_size, head_hidden_size, self.out_size, dropout
        )

        MB = mmmt.models.model_builder.ModelBuilder(
            graph_model, head_model, self.batch_size
        )
        h = MB.forward(self.graph_dict)
        self.assertEqual(h.shape[1], self.out_size)

    def tearDown(self):
        """Tear down the tests."""
        pass
