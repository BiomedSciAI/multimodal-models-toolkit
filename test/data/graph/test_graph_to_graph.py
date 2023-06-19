"""Unit test for mmmt.data.graph.graph2graph"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
import unittest
from mmmt.data.graph.graph_to_graph import GraphTransform
import numpy as np
import dgl


class GraphTransformTestCase(unittest.TestCase):
    """GraphTransformTestCase class."""

    def setUp(self):
        """Setting up the test."""
        # preparing dummy data
        self.num_nodes = 10
        self.num_edges = 50
        self.graph = dgl.rand_graph(self.num_nodes, self.num_edges)
        self.multi_graph_data = {
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
        self.multigraph = dgl.heterograph(self.multi_graph_data)

        self.node_features = np.random.rand(self.num_nodes)

        self.non_existing_method = "test"

    def test_mplex(self):
        """Test transformation to multiplex graph."""
        GT = GraphTransform(self.multigraph, self.node_features, "mplex")
        derived_graphs, mplex_node_features = GT.transform()
        graph_mplex_i, graph_mplex_ii = derived_graphs
        total_num_edges = 0
        for etype in graph_mplex_i.etypes:
            total_num_edges += graph_mplex_i.num_edges(etype)

        self.assertIsInstance(graph_mplex_i, dgl.DGLGraph)
        self.assertEqual(
            graph_mplex_i.num_nodes() * len(graph_mplex_i.etypes),
            self.num_nodes * len(self.multi_graph_data),
        )
        self.assertEqual(
            total_num_edges, self.num_edges * len(self.multi_graph_data) ** 2
        )

        self.assertIsInstance(graph_mplex_ii, dgl.DGLGraph)
        self.assertEqual(
            graph_mplex_ii.num_nodes() * len(graph_mplex_ii.etypes),
            self.num_nodes * len(self.multi_graph_data),
        )

    def test_mplex_prop(self):
        """Test transformation to multiplex graph with properties."""
        GT = GraphTransform(
            self.multigraph, self.node_features, "mplex-prop", alpha=0.3
        )
        derived_graphs, mplex_node_features = GT.transform()
        graph_mplex_i, graph_mplex_ii = derived_graphs
        total_num_edges = 0
        for etype in graph_mplex_i.etypes:
            total_num_edges += graph_mplex_i.num_edges(etype)
        ef_1 = graph_mplex_i.edata["w"].float()
        ef_2 = graph_mplex_ii.edata["w"].float()

        self.assertIsInstance(graph_mplex_i, dgl.DGLGraph)
        self.assertEqual(
            graph_mplex_i.num_nodes() * len(graph_mplex_i.etypes),
            self.num_nodes * len(self.multi_graph_data),
        )
        self.assertEqual(
            total_num_edges, self.num_edges * len(self.multi_graph_data) ** 2
        )

        self.assertIsInstance(graph_mplex_ii, dgl.DGLGraph)
        self.assertEqual(
            graph_mplex_ii.num_nodes() * len(graph_mplex_ii.etypes),
            self.num_nodes * len(self.multi_graph_data),
        )

        self.assertEqual(len(ef_1), self.num_edges * len(self.multi_graph_data) ** 2)
        self.assertEqual(len(ef_2), self.num_edges * len(self.multi_graph_data) ** 2)

    def test_multibehav(self):
        """Test transformation to mulit-behavioural graph."""
        GT = GraphTransform(self.multigraph, self.node_features, "multibehav")
        derived_graphs, mplex_node_features = GT.transform()
        quotient_graph, mplex_graph = derived_graphs

        self.assertIsInstance(quotient_graph, dgl.DGLGraph)
        self.assertEqual(
            quotient_graph.num_nodes() * len(quotient_graph.etypes), self.num_nodes
        )

        self.assertIsInstance(mplex_graph, dgl.DGLGraph)
        self.assertEqual(
            mplex_graph.num_nodes() * len(mplex_graph.etypes),
            self.num_nodes * len(self.multi_graph_data),
        )
        self.assertEqual(
            mplex_graph.num_edges(),
            self.num_edges * len(self.multi_graph_data)
            + self.num_nodes * len(self.multi_graph_data) * 2,
        )

    def test_mGNN(self):
        """Test transformation to mGNN."""
        GT = GraphTransform(self.multigraph, self.node_features, "mgnn")
        derived_graph, mplex_node_features = GT.transform()
        g_inter_layer, g_intra_layer = derived_graph

        self.assertIsInstance(g_inter_layer, dgl.DGLGraph)
        self.assertEqual(
            self.num_nodes * len(self.multi_graph_data), g_inter_layer.num_nodes()
        )
        self.assertEqual(
            self.num_nodes * len(self.multi_graph_data) * 2, g_inter_layer.num_edges()
        )

        self.assertIsInstance(g_intra_layer, dgl.DGLGraph)
        self.assertEqual(
            self.num_nodes * len(self.multi_graph_data), g_intra_layer.num_nodes()
        )
        self.assertEqual(
            self.num_edges * len(self.multi_graph_data), g_intra_layer.num_edges()
        )

    def test_non_existing_method(self):
        """Test that a non supported dataset raises an error"""
        with self.assertRaises(ValueError):
            GraphTransform(
                self.graph, self.node_features, self.non_existing_method
            ).transform()

    def tearDown(self):
        """Tear down the tests."""
        pass
