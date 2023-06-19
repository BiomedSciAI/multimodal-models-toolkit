"""Unit test for complex_module.core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
import os
from tempfile import mkdtemp
from mmmt.data.representation.modality_encoding import (
    ModalityEncoding,
)
import unittest
import numpy as np
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_read import OpReadDataframe
import pandas as pd


class ModalityEncodingTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        root = mkdtemp(prefix="METestCase")

        data = {
            "sample_id": [0, 1, 2, 3],
            "modality1": [
                np.random.rand(64).astype(np.float32),
                np.random.rand(64).astype(np.float32),
                np.random.rand(64).astype(np.float32),
                np.random.rand(64).astype(np.float32),
            ],
            "modality2": [
                np.random.rand(16).astype(np.float32),
                np.random.rand(16).astype(np.float32),
                np.random.rand(16).astype(np.float32),
                np.random.rand(16).astype(np.float32),
            ],
        }
        df = pd.DataFrame(data)

        pipeline_list = [
            (OpReadDataframe(df), dict()),
        ]
        dataset_pipeline = PipelineDefault("static", pipeline_list)

        # Define splits
        training_sample_ids = [0, 1]
        val_sample_ids = [2]
        test_sample_ids = [3]

        self.encoding_strategy = {
            "pipeline": {
                "fuse_pipeline": dataset_pipeline,
                "data_splits": {
                    "train_ids": training_sample_ids,
                    "val_ids": val_sample_ids,
                    "test_ids": test_sample_ids,
                },
            },
            "num_workers": 1,
            "restart_cache": True,
            "root_dir": root,
            "step_args": {
                "modality1": {
                    "model": None,
                    "use_autoencoder": False,
                    "output_key": "modality1_output_key",
                },
                "modality2": {
                    "model": None,
                    "add_feature_names": False,
                    "use_autoencoder": True,
                    "output_key": "modality1_output_key",
                    "encoding_layers": [16, 4],
                    "use_pretrained": False,
                    "batch_size": 3,
                    "training": {
                        "model_dir": "model_image_features",
                        "pl_trainer_num_epochs": 1,
                        "pl_trainer_accelerator": "cpu",
                    },
                },
            },
            "object_registry": None,
        }

        pass

    def test_empty_contructor(self):
        """Test salutation()."""
        with self.assertRaises(TypeError):
            ModalityEncoding()

    def test_pipeline_too_soon(self):
        """Test salutation()."""
        ME = ModalityEncoding(self.encoding_strategy)

        with self.assertRaises(AttributeError):
            ME.get_pipeline()

    def test_me(self):

        ME = ModalityEncoding(self.encoding_strategy)

        ME.__call__()

    def tearDown(self):
        """Tear down the tests."""
        pass


if __name__ == "__main__":
    unittest.main()
