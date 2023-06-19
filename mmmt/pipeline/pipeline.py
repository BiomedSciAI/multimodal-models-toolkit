import logging
import os
import re
from copy import copy

import mlflow
import yaml

from mmmt.pipeline.object_registry import ObjectRegistry

# add the option to extract values from env varaibles
env_pattern = re.compile(r".*?\${(.*?)}.*?")


def env_constructor(loader, node):
    value = loader.construct_scalar(node)
    for group in env_pattern.findall(value):
        if group not in os.environ or os.environ[group] is None:
            raise Exception(f"Error: missing env var {group}")
        print(f"Configuration file: env variable read {group}={os.environ.get(group)}")
        value = value.replace(f"${{{group}}}", os.environ.get(group))
    return value


yaml.add_implicit_resolver("!pathex", env_pattern)
yaml.add_constructor("!pathex", env_constructor)

logging.basicConfig(level=logging.INFO)


class MMMTPipeline:
    """
    MMMT pipeline, used to interpret the configuration set by the user. Values not defined by the user are taken from the defaults.yaml.

    A typical pipeline involves the following steps (as described in D'Souza et al. [1]_):

        1. Each modality is encoded from its raw representation into feature vectors using
        the relevant key in `encoding_strategy`. It allows to use a pretrained model, train a
        modality-specific autoencoder, a combination of model + autoencoder or none (raw
        representation needs to be a feature vector)

        2. All the encoded modalities are combined together with a `concept_encoder` that is a
        simple autoencoder that projects the modalities into a small latent space (autoencoder
        bottleneck).

        3. The unimodal embedded features are organized as a graph, where the links are obtained
        through detecting saliency of the features on each individual dimension of the concept
        embedding space (one edge type per dimension).
        Like this, the `data.base_graph` key will contain, for each sample, a
        representation of the unimodal embedded features.

        4. If the graph_module to be used is supporting multiplexed graphs, each of the edge types
        defines a graph layer, and nodes will be replicated across all layers and interlayer links
        are added into the `data.derived_graph` key.

        5. The samples transformed in graphs are used to train, validate and test a GNN as specified
        in the configuration yaml file.


        .. rubric:: References
        .. [1] D'Souza, Niharika, et al. "Fusing Modalities by Multiplexed Graph Neural Networks for Outcome Prediction in Tuberculosis." International Conference on Medical Image Computing and Computer-Assisted Intervention. 2022.
    """

    def __init__(
        self,
        user_configs_pipeline_path,
        specific_objects,
        defaults="mmmt/pipeline/defaults.yaml",
    ):
        """

        Args:
            user_configs_pipeline_path (str): path to case-specific configuration
            specific_objects (dict): dictionary of case-specific objects
        """

        self.obj_reg = ObjectRegistry(specific_objects)

        defaults_mmmt_pipeline = yaml.full_load(open(defaults, "r"))

        user_configs = yaml.full_load(open(user_configs_pipeline_path, "r"))

        self.mmmt_pipeline_config = self.update_config(
            defaults_mmmt_pipeline, user_configs
        )

        self.mlflow_configurator()

        self.cache = self.mmmt_pipeline_config["cache"]

        self.pipeline = {}

        logging.info("MMMT pipeline initialized")

    def update_config(self, to_be_updated, user_update):
        """
        Recursive function to update the nested configuration dictionary.
        Args:
            to_be_updated (dict): dictionary to be updated
            user_update (dict): dictionary containing the updated values

        Returns:
            updated dictionary
        """
        for k, v in user_update.items():
            if isinstance(v, dict):
                to_be_updated[k] = self.update_config(to_be_updated.get(k, {}), v)
            elif isinstance(v, list):
                if k not in to_be_updated:
                    to_be_updated[k] = copy(v)
                else:
                    for ind, elem in enumerate(v):
                        if isinstance(elem, dict):
                            to_be_updated[k][ind] = self.update_config(
                                to_be_updated[k][ind], elem
                            )
                        else:
                            if ind == len(to_be_updated[k]):
                                to_be_updated[k].append(elem)
                            else:
                                to_be_updated[k][ind] = elem
            else:
                to_be_updated[k] = v
        return to_be_updated

    def mlflow_configurator(
        self,
    ):
        for mlflow_env_key in self.mmmt_pipeline_config["mlflow"]:
            if (
                mlflow_env_key not in os.environ
                and self.mmmt_pipeline_config["mlflow"][mlflow_env_key]
            ):
                os.environ[mlflow_env_key] = self.mmmt_pipeline_config["mlflow"][
                    mlflow_env_key
                ]

    def run_pipeline(self, debugging=False):
        """
        Run each pipeline step.
        Args:
            debugging (bool, optional): boolean to control the number of samples. If it is boolean only few samples are used. Defaults to False.
        """
        with mlflow.start_run(nested=True):
            for phase in self.mmmt_pipeline_config:
                if phase in ["cache", "mlflow"]:
                    continue
                for step in self.mmmt_pipeline_config[phase]:

                    self.process_step(step)

                logging.info(phase + " processed")

                # for debugging
                if phase == "data" and debugging:
                    self.pipeline["data_splits"]["train_ids"] = self.pipeline[
                        "data_splits"
                    ]["train_ids"][:3]
                    self.pipeline["data_splits"]["val_ids"] = self.pipeline[
                        "data_splits"
                    ]["val_ids"][:3]
                    self.pipeline["data_splits"]["test_ids"] = self.pipeline[
                        "data_splits"
                    ]["test_ids"][:3]

    def process_step(self, step):
        """
        Process a step taking the configuration, in particular it expect an object and its arguments.
        Args:
            step (dict): Configuration of the step to initialize and execute.
        """
        if "fuse_object" in step:
            if "fuse_pipeline" not in self.pipeline:
                self.pipeline["fuse_pipeline"] = {}
            if not self.pipeline["fuse_pipeline"]:
                self.pipeline["fuse_pipeline"] = self.obj_reg.instance_object(
                    step["fuse_object"], step["args"]
                )
            else:
                self.pipeline["fuse_pipeline"].extend(
                    (self.obj_reg.object_dict[step["fuse_object"]], step["args"])
                )
        else:
            need_cache = self.obj_reg.object_dict[step["object"]].get(
                "need_cache", False
            )
            need_pipeline = self.obj_reg.object_dict[step["object"]].get(
                "need_pipeline", False
            )
            need_object_registry = self.obj_reg.object_dict[step["object"]].get(
                "need_object_registry", False
            )
            need_call_method = self.obj_reg.object_dict[step["object"]].get(
                "need_call_method", False
            )

            if need_cache or need_pipeline or need_object_registry:
                step_args = {"step_args": step["args"]}

                if need_cache:
                    step_args.update(self.cache)
                if need_pipeline:
                    step_args["pipeline"] = self.pipeline
                if need_object_registry:
                    step_args["object_registry"] = self.obj_reg
            else:
                step_args = step["args"]

            if "io" in step:

                self.pipeline[step["io"]["output_key"]] = self.obj_reg.instance_object(
                    step["object"], step_args
                )
                if need_call_method:
                    self.pipeline[step["io"]["output_key"]]()
            else:
                self.pipeline[step["object"]] = self.obj_reg.instance_object(
                    step["object"], step_args
                )
                if need_call_method:
                    self.pipeline[step["object"]]()
