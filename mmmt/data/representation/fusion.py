from abc import abstractmethod
from collections import Iterable

import torch
from fuse.data.datasets.dataset_base import DatasetDefault
from fuse.data.ops.ops_common import OpConcat
from fuse.data.pipelines.pipeline_default import PipelineDefault

from mmmt.data.operators.op_forwardpass import OpForwardPass


class FusionBaseClass:
    """
    Fusion base class.

    """

    def __init__(self, input_modality_keys: Iterable, output_key: str, **kwargs):
        super().__init__()
        self.input_modality_keys = input_modality_keys
        self.output_key = output_key

    def train(
        self,
        dataset: DatasetDefault = None,
        **kwargs,
    ) -> None:
        """trains/fits the fusion method on an user specified dataset.

        :param dataset: fuse dataset on which to train the fusion method, defaults to None
        :type dataset: DatasetDefault, optional
        """
        pass

    @abstractmethod
    def toNonDifferentiableFusePipeline(
        self, dataset_pipeline: PipelineDefault, **kwargs
    ) -> PipelineDefault:
        """provides the trained method as a fuse operator by extending a fuse pipeline object

        :param dataset_pipeline: data pipeline to be extended
        :type dataset_pipeline: PipelineDefault
        :return: Extended pipeline with this fusion operator
        :rtype: PipelineDefault
        """
        raise NotImplementedError


class DifferentiableFusion(FusionBaseClass):
    """
    Differentiable fusion class
    """

    def __init__(
        self,
        input_modality_keys: Iterable,
        output_key: str,
        modality_dimensions: int,
        **kwargs,
    ):
        super().__init__()
        self.modality_dimensions = modality_dimensions

    @abstractmethod
    def toPytorchModule(self) -> torch.nn.Module:
        """returns a pytorch module as differentiable fusion method

        :return: pytorch module so that it can be composed with others
        :rtype: torch.nn.Module
        """
        raise NotImplementedError

    def toNonDifferentiableFusePipeline(
        self, dataset_pipeline: PipelineDefault, **kwargs
    ) -> PipelineDefault:
        """provides the trained method as non differentiable a fuse operator by extending a
        fuse pipeline object. Subclasses might overwrite this method. By default it will concatenate
        input_modality_keys and feed a forward pass of the trained model

        :param dataset_pipeline: data pipeline to be extended
        :type dataset_pipeline: PipelineDefault
        :return: Extended pipeline with this fusion operator
        :rtype: PipelineDefault
        """
        self.concat_key = self.output_key + "_concat"
        dataset_pipeline.extend(
            [
                (
                    OpConcat(),
                    dict(
                        keys_in=self.input_modality_keys,
                        key_out=self.concat_key,
                        axis=0,
                    ),
                ),
                (
                    OpForwardPass(
                        self.toPytorchModule(),
                        modality_dimensions=self.modality_dimensions,
                        add_feature_names=False,
                    ),
                    dict(key_in=self.concat_key, key_out=self.output_key),
                ),
            ]
        )
