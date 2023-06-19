"""Unit test for module configurator for the models defined in mmmt.models.graph"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
import unittest
import dgl
import numpy as np
import torch
import copy

from mmmt.data.graph.graph_to_graph import GraphTransform
from mmmt.models.head.mlp import MLP
from mmmt.models.graph.module_configurator import ModuleConfigurator


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        # preparing dummy data
        self.non_existing_method = "test"

        # graph
        self.num_nodes = 10
        self.num_edges = 100
        self.graph = dgl.rand_graph(self.num_nodes, self.num_edges)
        multi_graph_data = {
            ("feat", "mod1", "feat"): dgl.rand_graph(
                self.num_nodes, self.num_edges
            ).edges(),
            ("feat", "mod2", "feat"): dgl.rand_graph(
                self.num_nodes, self.num_edges
            ).edges(),
            ("feat", "mod3", "feat"): dgl.rand_graph(
                self.num_nodes, self.num_edges
            ).edges(),
        }
        self.multigraph = dgl.heterograph(multi_graph_data)
        self.n_layers = len(self.multigraph.etypes)

        # node features
        self.node_emb_dim = 4
        self.node_features = np.random.rand(self.num_nodes, self.node_emb_dim).astype(
            np.float32
        )

        # head hyperparameters
        self.head_hidden_size = [10, 10]
        self.out_size = 5
        self.dropout = 0.5

        self.batch_size = 1

        self.constant_graph_module = {
            "thresh_q": 0.95,
            "node_emb_dim": self.node_emb_dim,
            "n_layers": self.n_layers,
        }

    def test_module_configurator_mplx_gin(self):
        module_identifier = "mplex"
        GT = GraphTransform(
            self.multigraph, self.node_features, graph_module=module_identifier
        )
        derived_graph, features_multigraph = GT.transform()
        graph_dict_sample = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        graph_module = copy.deepcopy(self.constant_graph_module)
        graph_module["module_identifier"] = module_identifier

        MC = ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(
            graph_dict_sample["graph"]
        )

        head_model = MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
        )

        h = graph_model.forward(graph_dict_sample)

        self.assertEqual(head_model.in_size, graph_model.out_size)
        self.assertEqual(head_model.in_size, h.shape[1])
        self.assertEqual(head_num_nodes, h.shape[0])

    def test_module_configurator_mplx_gcn(self):
        module_identifier = "mplex-prop"
        GT = GraphTransform(
            self.multigraph, self.node_features, graph_module=module_identifier
        )
        derived_graph, features_multigraph = GT.transform()
        graph_dict_sample = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        graph_module = copy.deepcopy(self.constant_graph_module)
        graph_module["module_identifier"] = module_identifier
        graph_module["gl_hidden_size"] = [2, 2]

        MC = ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(
            graph_dict_sample["graph"]
        )

        head_model = MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
        )

        h = graph_model.forward(graph_dict_sample)

        self.assertEqual(head_model.in_size, graph_model.out_size)
        self.assertEqual(head_model.in_size, h.shape[1])
        self.assertEqual(head_num_nodes, h.shape[0])

    def test_module_configurator_rgcn(self):
        module_identifier = "rgcn"
        GT = GraphTransform(
            self.multigraph, self.node_features, graph_module=module_identifier
        )
        derived_graph, features_multigraph = GT.transform()
        graph_dict_sample = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        graph_module = copy.deepcopy(self.constant_graph_module)
        graph_module["module_identifier"] = module_identifier
        graph_module["gl_hidden_size"] = [2]
        graph_module["num_bases"] = 8

        MC = ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(
            graph_dict_sample["graph"]
        )

        head_model = MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
        )

        h = graph_model.forward(graph_dict_sample)

        self.assertEqual(head_model.in_size, graph_model.out_size)
        self.assertEqual(head_model.in_size, h.shape[1])
        self.assertEqual(head_num_nodes, h.shape[0])

    def test_module_configurator_mgnn(self):
        module_identifier = "mgnn"
        GT = GraphTransform(
            self.multigraph, self.node_features, graph_module=module_identifier
        )
        derived_graph, features_multigraph = GT.transform()
        graph_dict_sample = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        graph_module = copy.deepcopy(self.constant_graph_module)
        graph_module["module_identifier"] = module_identifier
        graph_module["gl_hidden_size"] = [2]
        graph_module["num_att_heads"] = 8

        MC = ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(
            graph_dict_sample["graph"]
        )

        head_model = MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
        )

        h = graph_model.forward(graph_dict_sample)

        self.assertEqual(head_model.in_size, graph_model.out_size)
        self.assertEqual(head_model.in_size, h.shape[1])
        self.assertEqual(head_num_nodes, h.shape[0])

    def test_module_configurator_gcn(self):

        module_identifier = "gcn"
        GT = GraphTransform(
            self.multigraph, self.node_features, graph_module=module_identifier
        )
        derived_graph, features_multigraph = GT.transform()
        graph_dict_sample = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        graph_module = copy.deepcopy(self.constant_graph_module)
        graph_module["module_identifier"] = module_identifier
        graph_module["gl_hidden_size"] = [2]

        MC = ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(
            graph_dict_sample["graph"]
        )

        head_model = MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
        )

        h = graph_model.forward(graph_dict_sample)

        self.assertEqual(head_model.in_size, graph_model.out_size)
        self.assertEqual(head_model.in_size, h.shape[1])
        self.assertEqual(head_num_nodes, h.shape[0])

    def test_module_configurator_MBGCN(self):
        module_identifier = "multibehav"
        GT = GraphTransform(
            self.multigraph, self.node_features, graph_module=module_identifier
        )
        derived_graph, features_multigraph = GT.transform()
        graph_dict_sample = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }
        graph_module = copy.deepcopy(self.constant_graph_module)
        graph_module["module_identifier"] = module_identifier
        graph_module["gl_hidden_size"] = [2]

        MC = ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(
            graph_dict_sample["graph"]
        )

        head_model = MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
        )

        h = graph_model.forward(graph_dict_sample)

        self.assertEqual(head_model.in_size, graph_model.out_size)
        self.assertEqual(head_model.in_size, h.shape[1])
        self.assertEqual(head_num_nodes, h.shape[0])

    def test_module_configurator_non_existing_method(self):
        """Test that a non supported dataset raises an error"""
        graph_module = {
            "module_identifier": self.non_existing_method,
        }
        with self.assertRaises(ValueError):
            ModuleConfigurator(graph_module).get_module(None)

    def tearDown(self):
        """Tear down the tests."""
        pass
