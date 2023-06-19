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
from fuse.data.ops.ops_common import OpToOneHot
import pandas as pd
from mmmt.pipeline.pipeline import MMMTPipeline


class TESTDataset:
    @staticmethod
    def static_pipeline(name=None) -> PipelineDefault:
        data = {
            "sample_id": [i for i in range(100)],
            "data.input.raw.modality1": [
                np.random.rand(64).astype(np.float32) * (i % 2) for i in range(100)
            ],
            "data.input.raw.modality2": [
                np.random.rand(32).astype(np.float32) * (i % 2) for i in range(100)
            ],
            "data.input.raw.modality3": [
                np.random.rand(128).astype(np.float32) * (i % 2) for i in range(100)
            ],
            "data.ground_truth": [np.random.randint(2, size=1) for i in range(100)],
        }

        df = pd.DataFrame(data)
        df["data.ground_truth"] = df.applymap(
            lambda x: int(2 * np.mean(x)), na_action="ignore"
        )["data.input.raw.modality1"]

        pipeline_list = [
            (OpReadDataframe(df), dict()),
            (
                OpToOneHot(2),
                {"key_in": "data.ground_truth", "key_out": "data.ground_truth"},
            ),
        ]
        dataset_pipeline = PipelineDefault("static", pipeline_list)
        print(name)
        return dataset_pipeline

    @staticmethod
    def get_splits(name=None):

        data_splits = {
            "train_ids": list(range(50)),
            "val_ids": list(range(50, 75)),
            "test_ids": list(range(75, 100)),
        }
        print(name)
        return data_splits


class PipelineTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""

        self.specific_objects = {
            "static_pipeline": {
                "object": TESTDataset.static_pipeline,
            },
            "get_splits": {
                "object": TESTDataset.get_splits,
            },
        }

        pass

    def test_create_pipeline(self):
        """Test cteating a pipeline."""
        MMMTP = MMMTPipeline(
            "test/pipeline/test.yaml",
            self.specific_objects,
            defaults="test/pipeline/test.yaml",
        )

    def test_run_pipeline(self):
        """Test running the pipeline."""
        MMMTP = MMMTPipeline(
            "test/pipeline/test.yaml",
            self.specific_objects,
            defaults="test/pipeline/test.yaml",
        )
        MMMTP.run_pipeline()

    def test_run_mlp_pipeline(self):
        """Test running the pipeline."""
        MMMTP = MMMTPipeline(
            "test/pipeline/test_mlp.yaml",
            self.specific_objects,
            defaults="test/pipeline/test_mlp.yaml",
        )
        MMMTP.run_pipeline()

    def tearDown(self):
        """Tear down the tests."""
        pass


if __name__ == "__main__":
    unittest.main()
