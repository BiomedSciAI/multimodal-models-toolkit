"""Unit test for mmmt.models.multimodal_graph_model"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

from tempfile import mkdtemp
import os
import itertools
import numpy as np
import pandas as pd
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.utils.collates import CollateDefault
from torch.utils.data.dataloader import DataLoader
import unittest
import dgl
import torch
from fuse.data.datasets.dataset_default import DatasetDefault
from mmmt.data.graph.graph_to_graph import GraphTransform
from mmmt.models.multimodal_graph_model import (
    MultimodalGraphModel,
    custom_collate_graph,
    custom_collate_tensor,
)
from mmmt.pipeline.object_registry import ObjectRegistry


class MGMTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        num_nodes = 5
        num_edges = 10
        self.graph = dgl.rand_graph(num_nodes, num_edges)
        self.node_emb_dim = 2
        self.out_size = 5
        self.features = np.random.rand(num_nodes, self.node_emb_dim).astype(np.float32)
        self.root = mkdtemp(prefix="MGMTestCase")
        self.training_sample_ids = None
        self.val_sample_ids = None
        self.test_sample_ids = None
        self.graph_training_batch_sizes = [1, 2]
        self.validation_metrics = [
            ("accuracy", "MetricAccuracy"),
            ("auc", "MetricAUCROC"),
        ]

    def prepare_testing(
        self,
        graph_model_input,
        validation_metric_key,
        validation_metric,
        graph_training_batch_size,
    ):
        GT = GraphTransform(
            self.graph,
            self.features,
            graph_module=graph_model_input["module_identifier"],
        )
        derived_graph, features_multigraph = GT.transform()

        graph_data = {
            "sample_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "derived_graph": [
                {
                    "graph": derived_graph,
                    "node_features": features_multigraph,
                }
            ]
            * 10,
            "label": [
                0,
                1,
                2,
                2,
                0,
                4,
                3,
                2,
                1,
                4,
            ],
        }
        df = pd.DataFrame(graph_data)

        pipeline_list = [
            (OpReadDataframe(df), dict()),
            (OpToNumpy(), dict(key="derived_graph.node_features")),
        ]
        dataset_pipeline = PipelineDefault("static", pipeline_list)

        self.training_sample_ids = [0, 1, 2]
        self.val_sample_ids = [3, 4, 5]
        self.test_sample_ids = [6, 7, 8]

        training_dataset = DatasetDefault(
            sample_ids=self.training_sample_ids,
            static_pipeline=dataset_pipeline,
        )
        training_dataset.create()

        validation_dataset = DatasetDefault(
            sample_ids=self.val_sample_ids,
            static_pipeline=dataset_pipeline,
        )
        validation_dataset.create()

        test_dataset = DatasetDefault(
            sample_ids=self.test_sample_ids,
            static_pipeline=dataset_pipeline,
        )
        test_dataset.create()

        pipeline = {
            "fused_dataset": {
                "graph_train_dataset": training_dataset,
                "graph_validation_dataset": validation_dataset,
                "graph_test_dataset": test_dataset,
            }
        }

        args_dict = {
            "pipeline": pipeline,
            "step_args": {
                "model_config": {
                    "graph_model": graph_model_input,
                    "head_model": {
                        "head_hidden_size": [10, 5],
                        "dropout": 0.5,
                        "add_softmax": True,
                    },
                    "num_classes": self.out_size,
                },
                "training": {
                    "model_dir": "model_mplex",
                    "batch_size": graph_training_batch_size,
                    "best_epoch_source": {
                        "mode": "max",
                        "monitor": "validation.metrics." + validation_metric_key,
                    },
                    "train_metrics": {
                        "key": "auc",
                        "object": "MetricAUCROC",
                        "args": {
                            "pred": "model.out",
                            "target": "label",
                        },
                    },
                    "validation_metrics": {
                        "key": validation_metric_key,
                        "object": validation_metric,
                        "args": {
                            "pred": "model.out",
                            "target": "label",
                        },
                    },
                    "pl_trainer_num_epochs": 2,
                    "pl_trainer_accelerator": "cpu",
                    "pl_trainer_devices": 1,
                },
                "testing": {
                    "test_results_filename": "test_results.pickle",
                    "evaluation_directory": "eval",
                },
                "io": {
                    "fused_dataset_key": "fused_dataset",
                    "input_key": "derived_graph",
                    "target_key": "label",
                    "prediction_key": "model.out",
                },
            },
            "root_dir": self.root,
            "num_workers": 1,
            "object_registry": ObjectRegistry(),
        }

        MGM = MultimodalGraphModel(args_dict)

        MGM.train()

        test_df = MGM.test()

        return MGM, test_df

    def test_empty_contructor(self):
        """Test empty arguments."""
        with self.assertRaises(TypeError):
            MultimodalGraphModel()

    def test_wrong_parameters(self):
        """Test wrong arguments."""

        with self.assertRaises(TypeError):
            MultimodalGraphModel([])

    def test_forward_pass(self):

        """Test training and model forward pass."""

        graph_model_input = {
            "module_identifier": "mplex",
            "node_emb_dim": self.node_emb_dim,
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            MGM, _ = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            trained_model = MGM.mb_trainer.load_checkpoint(
                os.path.join(MGM.model_dir, MGM.checkpoint_filename)
            )

            graph_testing_dataloader = DataLoader(
                dataset=MGM.graph_test_dataset,
                batch_size=graph_training_batch_size,
                collate_fn=CollateDefault(
                    special_handlers_keys={
                        MGM.graph_key + ".graph": custom_collate_graph,
                        MGM.graph_key + ".node_features": custom_collate_tensor,
                    }
                ),
                num_workers=MGM.num_workers,
            )

            for batch in graph_testing_dataloader:
                batch_forward = trained_model.forward(batch)
                prediction = batch_forward["model"]["out"]

                self.assertEqual(
                    prediction.shape[1],
                    self.out_size,
                )

    def test_mgm_mplex(self):
        """Test training and model forward pass."""

        graph_model_input = {
            "module_identifier": "mplex",
            "node_emb_dim": self.node_emb_dim,
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            _, test_df = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            self.assertEqual(
                len(test_df["pred"][0]),
                self.out_size,
            )

            self.assertEqual(
                len(test_df),
                len(self.test_sample_ids),
            )

    def test_mgm_mplex_prop(self):
        graph_model_input = {
            "module_identifier": "mplex-prop",
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": [2, 2],
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            _, test_df = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            self.assertEqual(
                len(test_df["pred"][0]),
                self.out_size,
            )

            self.assertEqual(
                len(test_df),
                len(self.test_sample_ids),
            )

    def test_mgm_gcn(self):
        graph_model_input = {
            "module_identifier": "gcn",
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": [2],
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            _, test_df = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            self.assertEqual(
                len(test_df["pred"][0]),
                self.out_size,
            )

            self.assertEqual(
                len(test_df),
                len(self.test_sample_ids),
            )

    def test_mgm_mgnn(self):
        graph_model_input = {
            "module_identifier": "mgnn",
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": [2],
            "num_att_heads": 4,
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            _, test_df = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            self.assertEqual(
                len(test_df["pred"][0]),
                self.out_size,
            )

            self.assertEqual(
                len(test_df),
                len(self.test_sample_ids),
            )

    def test_mgm_multibehav(self):
        graph_model_input = {
            "module_identifier": "multibehav",
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": [2],
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            _, test_df = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            self.assertEqual(
                len(test_df["pred"][0]),
                self.out_size,
            )

            self.assertEqual(
                len(test_df),
                len(self.test_sample_ids),
            )

    def test_mgm_rgcn(self):
        graph_model_input = {
            "module_identifier": "rgcn",
            "node_emb_dim": self.node_emb_dim,
            "gl_hidden_size": [2],
            "num_bases": 8,
            "n_layers": 1,
        }

        for validation_metrics, graph_training_batch_size in itertools.product(
            self.validation_metrics, self.graph_training_batch_sizes
        ):

            validation_metric_key, validation_metric = validation_metrics

            _, test_df = self.prepare_testing(
                graph_model_input,
                validation_metric_key,
                validation_metric,
                graph_training_batch_size,
            )

            self.assertEqual(
                len(test_df["pred"][0]),
                self.out_size,
            )

            self.assertEqual(
                len(test_df),
                len(self.test_sample_ids),
            )

    def tearDown(self):
        """Tear down the tests."""
        pass


if __name__ == "__main__":
    unittest.main()
