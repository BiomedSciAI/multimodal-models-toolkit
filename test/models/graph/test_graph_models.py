"""Unit test for models defined in mmmt.models.graph"""

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

from mmmt.models.graph.multiplex_gin import MultiplexGIN
from mmmt.models.graph.multiplex_gcn import MultiplexGCN
from mmmt.data.graph.graph_to_graph import GraphTransform
from mmmt.models.graph.relational_gcn import Relational_GCN
from mmmt.models.graph.mgnn import mGNN
from mmmt.models.graph.gcn import GCN
from mmmt.models.graph.multi_behavioral_gnn import MultiBehavioralGNN


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        # preparing dummy data
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

        self.node_emb_dim = 4
        self.features = np.random.rand(self.num_nodes, self.node_emb_dim).astype(
            np.float32
        )

    def test_mplx_gin(self):
        GT = GraphTransform(self.multigraph, self.features, graph_module="mplex")
        derived_graph, features_multigraph = GT.transform()
        graph_dict = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }

        module_input = {
            "node_emb_dim": self.node_emb_dim,
        }

        MPLX_GIN = MultiplexGIN(module_input)

        h = MPLX_GIN.forward(graph_dict)
        self.assertEqual(self.num_nodes * self.n_layers, h.shape[0])
        self.assertEqual(self.node_emb_dim * 4, h.shape[1])

    def test_mplx_gcn(self):
        GT = GraphTransform(self.multigraph, self.features, graph_module="mplex-prop")
        derived_graph, features_multigraph = GT.transform()
        graph_dict = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }

        in_size = self.node_emb_dim
        gl_hidden_size = [max(int(in_size / 2), 1), 5]
        module_input = {
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": gl_hidden_size,
        }

        MPLX_GCN = MultiplexGCN(module_input)

        h = MPLX_GCN.forward(graph_dict)
        self.assertEqual(self.num_nodes * self.n_layers, h.shape[0])
        self.assertEqual(gl_hidden_size[-1] * 2, h.shape[1])

    def test_rgcn(self):
        GT = GraphTransform(self.multigraph, self.features, graph_module="rgcn")
        graph, features_graph = GT.transform()
        graph_dict = {
            "graph": graph,
            "node_features": torch.from_numpy(features_graph),
        }

        in_size = self.node_emb_dim
        gl_hidden_size = [int(in_size / 2)]
        num_bases = 8
        module_input = {
            "node_emb_dim": self.node_emb_dim,
            "n_layers": self.n_layers,
            "num_bases": num_bases,
            "gl_hidden_size": gl_hidden_size,
        }

        RGCN = Relational_GCN(module_input)

        h = RGCN.forward(graph_dict)
        self.assertEqual(self.num_nodes, h.shape[0])
        self.assertEqual(gl_hidden_size[-1], h.shape[1])

    def test_mgnn(self):
        GT = GraphTransform(self.multigraph, self.features, graph_module="mgnn")
        derived_graph, features_multigraph = GT.transform()
        graph_dict = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(features_multigraph),
        }

        in_size = self.node_emb_dim
        gl_hidden_size = [int(in_size / 2)]
        num_att_heads = 8
        module_input = {
            "node_emb_dim": self.node_emb_dim,
            "n_layers": self.n_layers,
            "num_att_heads": num_att_heads,
            "gl_hidden_size": gl_hidden_size,
        }

        MGNN = mGNN(module_input)

        h = MGNN.forward(graph_dict)
        self.assertEqual(self.num_nodes * self.n_layers, h.shape[0])
        self.assertEqual(gl_hidden_size[-1] * 2 * num_att_heads, h.shape[1])
        # factor 2 above for concatenation of inter and intra layer

    def test_gcn(self):
        GT = GraphTransform(self.multigraph, self.features, graph_module="gcn")
        multigraph, features_multigraph = GT.transform()
        graph_dict = {
            "graph": multigraph,
            "node_features": torch.from_numpy(features_multigraph),
        }

        in_size = self.node_emb_dim
        gl_hidden_size = [int(in_size / 2)]
        module_input = {
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": gl_hidden_size,
        }

        GCN_obj = GCN(module_input)

        h = GCN_obj.forward(graph_dict)
        self.assertEqual(self.num_nodes, h.shape[0])
        self.assertEqual(gl_hidden_size[-1], h.shape[1])

    def test_MBGCN(self):
        GT = GraphTransform(self.multigraph, self.features, graph_module="multibehav")
        derived_graph, features_multigraph = GT.transform()
        graph_dict = {
            "graph": derived_graph,
            "node_features": torch.from_numpy(self.features),
        }

        in_size = self.node_emb_dim
        gl_hidden_size = [int(in_size / 2)]
        module_input = {
            "node_emb_dim": self.node_emb_dim,
            "n_layers": self.n_layers,
            "gl_hidden_size": gl_hidden_size,
        }

        MBGCN = MultiBehavioralGNN(module_input)

        h = MBGCN.forward(graph_dict)
        self.assertEqual(self.num_nodes * self.n_layers, h.shape[0])
        self.assertEqual(gl_hidden_size[-1], h.shape[1])

    def tearDown(self):
        """Tear down the tests."""
        pass
