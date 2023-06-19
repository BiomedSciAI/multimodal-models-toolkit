import os
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.utils.ndict import NDict

from mmmt.data.operators.op_build_graph import OpBuildBaseGraph, OpBuildDerivedGraph

import mlflow


class ConceptToGraph:
    """
    Transformation of samples in concatenated space to graphs using concepts
    """

    def __init__(
        self,
        args_dict,
    ):
        """
        Args:
            args_dict (dict): arguments of ConceptToGraph as defined in configuration yaml
        """

        self.training_sample_ids = args_dict["pipeline"]["data_splits"]["train_ids"]
        self.val_sample_ids = args_dict["pipeline"]["data_splits"]["val_ids"]
        self.test_sample_ids = args_dict["pipeline"]["data_splits"]["test_ids"]

        self.concept_encoder_model = args_dict["pipeline"][
            args_dict["step_args"]["io"]["concept_encoder_model_key"]
        ]

        self.thresh_q = args_dict["step_args"]["thresh_q"]
        self.graph_module = args_dict["step_args"]["module_identifier"]
        self.cache_graph_config = {
            "workers": args_dict["num_workers"],
            "restart_cache": args_dict["restart_cache"],
            "math_epsilon": 1e-5,
        }
        self.root_dir = args_dict["root_dir"]

        self.fused_dataset_key = args_dict["step_args"]["io"]["fused_dataset_key"]
        self.input_key = args_dict["step_args"]["io"]["input_key"]
        self.output_key = args_dict["step_args"]["io"]["output_key"]

        self.concept_to_graph_config = args_dict["step_args"]
        self.mmmt_pipeline = args_dict["pipeline"]
        self.pipeline = args_dict["pipeline"]["fuse_pipeline"]

    def __call__(
        self,
    ):
        """
        Build graphs from samples by extending the FuseMedML pipeline with specific operators

        Returns:
            train, validation and test sets with samples structured as graphs
        """
        with mlflow.start_run(run_name=f"{self.__class__.__qualname__}", nested=True):
            mlflow.log_params(NDict(self.concept_to_graph_config).flatten())
            mlflow.log_param("parent", "yes")
            self.pipeline.extend(
                [
                    (
                        OpBuildBaseGraph(self.concept_encoder_model, self.thresh_q),
                        dict(
                            key_in_concat=self.input_key,
                            key_in_concept="data.forward_pass.multimodal",
                            key_out="data.base_graph",
                        ),
                    ),
                    (
                        OpBuildDerivedGraph(self.graph_module),
                        dict(
                            key_in="data.base_graph",
                            key_out=self.output_key,
                        ),
                    ),
                ]
            )

            if self.cache_graph_config is None:
                self.cache_graph_config = {}
            if "cache_dirs" not in self.cache_graph_config:
                self.cache_graph_config["cache_dirs"] = [
                    os.path.join(self.root_dir, "cache_graph")
                ]

            cacher_graph = SamplesCacher(
                "cache_graph",
                self.pipeline,
                audit_first_sample=False,
                audit_rate=None,  # disabling audit because deepdiff returns an error when comparing tensors, which are contained in derived_graphs from modules ['gcn', 'rgcn', 'mplex-prop']
                **self.cache_graph_config,
            )
            graph_train_dataset = DatasetDefault(
                sample_ids=self.training_sample_ids,
                static_pipeline=self.pipeline,
                cacher=cacher_graph,
            )
            graph_train_dataset.create()

            graph_validation_dataset = DatasetDefault(
                sample_ids=self.val_sample_ids,
                static_pipeline=self.pipeline,
                cacher=cacher_graph,
            )
            graph_validation_dataset.create()

            if self.test_sample_ids is not None:
                graph_test_dataset = DatasetDefault(
                    sample_ids=self.test_sample_ids,
                    static_pipeline=self.pipeline,
                    cacher=cacher_graph,
                )
                graph_test_dataset.create()
            else:
                graph_test_dataset = None

            self.mmmt_pipeline[self.fused_dataset_key] = {
                "graph_train_dataset": graph_train_dataset,
                "graph_validation_dataset": graph_validation_dataset,
                "graph_test_dataset": graph_test_dataset,
            }
