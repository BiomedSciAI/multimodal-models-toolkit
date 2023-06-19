import os
import logging


import torch
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.ops.ops_cast import OpToNumpy, OpToTensor
from fuse.data.ops.ops_common import OpCond, OpLambda, OpDeleteKeypaths
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.collates import CollateDefault
from fuse.utils.ndict import NDict


from mmmt.data.operators.op_forwardpass import OpForwardPass

from torch.utils.data.dataloader import DataLoader

import mmmt
import mlflow
import mlflow.pytorch


class ModalityEncoding:
    """
    From raw unimodal data produces an encoding for each modality
    """

    def __init__(self, args_dict):
        """

        Args:
            args_dict (dict): arguments of ModalityEncoding as defined in configuration yaml
        """
        self.modality_encoding_strategy = args_dict["step_args"]

        self.training_sample_ids = args_dict["pipeline"]["data_splits"]["train_ids"]
        self.val_sample_ids = args_dict["pipeline"]["data_splits"]["val_ids"]
        self.test_sample_ids = args_dict["pipeline"]["data_splits"]["test_ids"]

        self.num_workers = args_dict["num_workers"]

        self.cache_unimodal_config = {
            "workers": args_dict["num_workers"],
            "restart_cache": args_dict["restart_cache"],
            "math_epsilon": 1e-5,
        }
        self.root_dir = args_dict["root_dir"]

        dataset_pipeline = args_dict["pipeline"]["fuse_pipeline"]

        if dataset_pipeline is None:
            self.pipeline = PipelineDefault("static", [])
        else:
            self.pipeline = dataset_pipeline

        self.obj_reg = args_dict["object_registry"]

    def __call__(
        self,
    ):

        """
        Encode modalities by training AEs if needed and passing the raw modalities to the corresponding encoders.
        Results are stored in the pipeline object.

        """

        # Loop over modalities to obtain feature vector representations with
        # provided pretrained models.
        feature_encoding = {}
        for modality_key, encoder in self.modality_encoding_strategy.items():
            logging.info(f"Setting up feature vector encoding for {modality_key}")
            if encoder is not None:
                if (
                    encoder.get("model_path") is not None
                    or encoder.get("model") is not None
                ):
                    if encoder.get("model") is not None:
                        model = self.obj_reg.instance_object(encoder["model"], {})
                    else:
                        model = torch.load(encoder["model_path"])
                    dimensions = encoder["dimensions"]
                    output_key = f"{modality_key}_features"
                    feature_encoding[modality_key] = output_key
                    self.pipeline.extend(
                        [
                            (OpToTensor(), dict(key=modality_key)),
                            (
                                OpForwardPass(
                                    model,
                                    dimensions,
                                    add_feature_names=encoder.get(
                                        "add_feature_names", True
                                    ),
                                ),
                                dict(
                                    key_in=modality_key,
                                    key_out=output_key,
                                ),
                            ),
                            (
                                OpCond(OpDeleteKeypaths()),
                                dict(
                                    condition=encoder.get("delete_modality", True),
                                    keypaths=[modality_key],
                                ),
                            ),
                            (OpLambda(torch.flatten), dict(key=output_key)),
                            (OpToNumpy(), dict(key=output_key)),
                        ]
                    )

        if self.cache_unimodal_config is None:
            self.cache_unimodal_config = {}
        if "cache_dirs" not in self.cache_unimodal_config:
            self.cache_unimodal_config["cache_dirs"] = [
                os.path.join(self.root_dir, "cache_unimodal")
            ]

        cacher_unimodal = SamplesCacher(
            "cache_unimodal",
            self.pipeline,
            **self.cache_unimodal_config,
        )

        training_dataset = DatasetDefault(
            sample_ids=self.training_sample_ids,
            static_pipeline=self.pipeline,
            cacher=cacher_unimodal,
        )
        training_dataset.create()

        val_dataset = DatasetDefault(
            sample_ids=self.val_sample_ids,
            static_pipeline=self.pipeline,
            cacher=cacher_unimodal,
        )
        val_dataset.create()

        # Training Autoencoders for each modality
        ae_encoders = {}
        ae_decoders = {}
        unimodal_rep = {}

        with mlflow.start_run(run_name=f"{self.__class__.__qualname__}", nested=True):
            mlflow.log_params(NDict(self.modality_encoding_strategy).flatten())
            mlflow.log_param("parent", "yes")
            for modality_key, encoder in self.modality_encoding_strategy.items():
                with mlflow.start_run(run_name=f"{modality_key}", nested=True):
                    mlflow.log_param("parent", "no")
                    mlflow.log_params(NDict(encoder).flatten())
                    logging.debug(f"{modality_key}: {encoder}")
                    if modality_key in feature_encoding:
                        data_key = feature_encoding[modality_key]
                    else:
                        data_key = modality_key
                    if encoder["use_autoencoder"]:
                        logging.info(
                            f"Setting up autoencoding for {modality_key} using {data_key}"
                        )
                        modality_feature_size = len(training_dataset[0][data_key])
                        encoding_layers = encoder.get("encoding_layers", [16])
                        autoencoder = mmmt.data.representation.AutoEncoderBuilder(
                            modality_feature_size, encoding_layers
                        )

                        ae_encoders[modality_key] = autoencoder.encoder()
                        logging.debug(ae_encoders[modality_key])
                        ae_decoders[modality_key] = autoencoder.decoder()
                        logging.debug(ae_decoders[modality_key])

                        unimodal_rep[
                            modality_key
                        ] = mmmt.data.representation.AutoEncoderTrainer(
                            ae_encoders[modality_key],
                            data_key,
                            f"model.embedding.{data_key}",
                            ae_decoders[modality_key],
                            f"model.reconstruction.{data_key}",
                        )
                        modality_train_config = self.modality_encoding_strategy[
                            modality_key
                        ].get("training", None)
                        if modality_train_config is None:
                            modality_train_config = {}

                        if "model_dir" not in modality_train_config:
                            modality_train_config["model_dir"] = os.path.join(
                                self.root_dir, "models", modality_key
                            )
                        else:
                            modality_train_config["model_dir"] = os.path.join(
                                self.root_dir,
                                modality_train_config["model_dir"],
                                modality_key,
                            )

                        unimodal_rep[modality_key].set_train_config(
                            **modality_train_config,
                        )
                        logging.info(f"Training autoencoder for {data_key}")

                        ckpt_file = os.path.join(
                            modality_train_config["model_dir"], "best_epoch.ckpt"
                        )
                        use_pretrained = encoder.get("use_pretrained", False)

                        if not use_pretrained or not os.path.exists(ckpt_file):
                            logging.debug(f"Pretrained: {use_pretrained}")
                            logging.debug(
                                f"{ckpt_file} exists: {os.path.exists(ckpt_file)}"
                            )
                            logging.info(f"Starting training of AE for {modality_key}")

                            # Create datasets for train and validation
                            batch_size = encoder["batch_size"]

                            # Unimodal representation
                            train_dataloader = DataLoader(
                                dataset=training_dataset,
                                batch_size=batch_size,
                                collate_fn=CollateDefault(),
                                num_workers=self.num_workers,
                            )

                            validation_dataloader = DataLoader(
                                dataset=val_dataset,
                                batch_size=batch_size,
                                collate_fn=CollateDefault(),
                                num_workers=self.num_workers,
                            )

                            unimodal_rep[modality_key].fit(
                                train_dataloader, validation_dataloader
                            )
                        else:
                            logging.info(
                                f"Training of AE for {modality_key} is skipped as trained model already exists in {os.path.exists(ckpt_file)}"
                            )
                        unimodal_rep[modality_key].load_checkpoint(ckpt_file)

                    else:
                        logging.info(
                            f"Skipping autoencoding for {modality_key} using {data_key}. Using Identity"
                        )
                        ae_encoders[modality_key] = torch.nn.Identity()
                        ae_decoders[modality_key] = torch.nn.Identity()

        concatenating_keys = []
        for modality_key, encoder in self.modality_encoding_strategy.items():
            if modality_key in feature_encoding:
                data_key = feature_encoding[modality_key]
            else:
                data_key = modality_key
            unimodal_rep_key = encoder[
                "output_key"
            ]  # no defaults here as if not defined should raise an error
            if encoder.get("add_feature_names", True):
                concatenating_keys.append(unimodal_rep_key)
            self.pipeline.extend(
                [
                    (
                        OpForwardPass(ae_encoders[modality_key], 1),
                        dict(
                            key_in=data_key,
                            key_out=unimodal_rep_key,
                        ),
                    ),
                    (OpToNumpy(), dict(key=unimodal_rep_key)),
                ]
            )
