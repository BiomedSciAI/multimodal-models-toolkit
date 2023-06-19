"""Unit test for complex_module.core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import unittest
import torch
import pandas as pd
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToTensor
from mmmt.data.operators.op_forwardpass import OpForwardPass
from mmmt.data.operators.op_concat_names import OpConcatNames
from mmmt.data.operators.op_resample import Op3DResample
import numpy as np


class GBMRTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""

        data = {
            "sample_id": [0, 1, 2, 3],
            "modality1": [
                np.random.randint((1, 5, 5)),
                np.random.randint((1, 5, 5)),
                np.random.randint((1, 5, 5)),
                np.random.randint((1, 5, 5)),
            ],
            "modality2": [
                np.random.randint((1, 5)),
                np.random.randint((1, 5)),
                np.random.randint((1, 5)),
                np.random.randint((1, 5)),
            ],
            "modality3D": [
                torch.rand((5, 5, 3)),
                torch.rand((5, 5, 3)),
                torch.rand((5, 5, 3)),
                torch.rand((5, 5, 3)),
            ],
        }
        self.df = pd.DataFrame(data)

        pass

    def test_resample(self):
        """Test resample()."""
        pipeline_list = [
            (OpReadDataframe(self.df), dict()),
            (OpToTensor(), dict(key="modality3D", dtype=torch.float)),
            (
                Op3DResample([4, 2, 3]),
                dict(key_in="modality3D", key_out="modality3D_resampled"),
            ),
        ]
        pipeline = PipelineDefault("test_mock_data", pipeline_list)
        sample_ids = list(range(4))

        dataset = DatasetDefault(sample_ids, static_pipeline=pipeline)
        dataset.create()
        assert torch.numel(dataset[0]["modality3D_resampled"]) == 24

    def test_forwardpass_and_concat_names(self):
        """Test forwardpass()."""
        identity = torch.nn.Identity()
        pipeline_list = [
            (OpReadDataframe(self.df), dict()),
            (
                OpForwardPass(identity, 1),
                dict(key_in="modality2", key_out="modality2fp"),
            ),
            (
                OpConcatNames(),
                dict(keys_in=["modality2fp", "modality2fp"], key_out="names.concat"),
            ),
        ]
        pipeline = PipelineDefault("test_mock_data", pipeline_list)
        sample_ids = list(range(4))

        dataset = DatasetDefault(sample_ids, static_pipeline=pipeline)
        dataset.create()

        assert len(dataset[0]["names.modality2fp"]) + len(
            dataset[0]["names.modality2fp"]
        ) == len(dataset[0]["names.concat"])

    def tearDown(self):
        """Tear down the tests."""
        pass
