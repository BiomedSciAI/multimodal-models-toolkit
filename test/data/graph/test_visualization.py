"""Unit test for complex_module.core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import unittest
from mmmt.data.graph.visualization import GraphVisualization
import dgl
from dgl import DGLGraph
import torch
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_read import OpReadDataframe


import pandas as pd


class GraphVisualizationTestCase(unittest.TestCase):
    """MatFileLoaderTestCase     class."""

    def setUp(self):
        """Setting up the test."""

        self.modality_names = (
            ["modality1.features"] * 2
            + ["modality2.features"] * 2
            + ["modality3.features"] * 2
        )

        data = {
            "sample_id": [0, 1, 2, 3],
            "node_names": [
                self.modality_names,
                self.modality_names,
                self.modality_names,
                self.modality_names,
            ],
            "graph": [
                dgl.rand_graph(6, 2),
                dgl.rand_graph(6, 1),
                dgl.rand_graph(6, 2),
                dgl.rand_graph(6, 4),
            ],
        }
        df = pd.DataFrame(data)
        pipeline = PipelineDefault("test_mock_data", [(OpReadDataframe(df), dict())])
        sample_ids = list(range(4))

        self.dataset = DatasetDefault(sample_ids, static_pipeline=pipeline)
        self.dataset.create()
        pass

    def test_visualization(self):
        """Test that visualization works."""
        G = GraphVisualization.visualize_dataset(
            self.dataset, graph_key="graph", node_names_key="node_names"
        )
        assert len(G.nodes) == 6

    def tearDown(self):
        """Tear down the tests."""
        pass
