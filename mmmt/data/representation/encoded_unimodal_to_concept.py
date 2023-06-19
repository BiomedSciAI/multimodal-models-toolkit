import os
import logging


from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.ops_common import OpConcat
from fuse.data.utils.collates import CollateDefault
from fuse.utils.ndict import NDict


from mmmt.data.operators.op_forwardpass import OpForwardPass
from mmmt.data.operators.op_concat_names import OpConcatNames

from torch.utils.data.dataloader import DataLoader

import mmmt
import mlflow


class EncodedUnimodalToConcept:
    """
    From encoding for each modality produces the concept encoder
    """

    def __init__(self, args_dict):
        """

        Args:
            args_dict (dict): arguments of EncodedUnimodalToConcept as defined in configuration yaml
        """
        self.concept_encoding_strategy = args_dict["step_args"]
        self.concept_train_config = args_dict["step_args"]["training"]

        self.training_sample_ids = args_dict["pipeline"]["data_splits"]["train_ids"]
        self.val_sample_ids = args_dict["pipeline"]["data_splits"]["val_ids"]
        self.test_sample_ids = args_dict["pipeline"]["data_splits"]["test_ids"]

        self.batch_size = args_dict["step_args"]["batch_size"]
        self.num_workers = args_dict["num_workers"]

        self.cache_concept_config = {
            "workers": args_dict["num_workers"],
            "restart_cache": args_dict["restart_cache"],
            "math_epsilon": 1e-5,
        }
        self.root_dir = args_dict["root_dir"]

        self.concept_encoder_model_key = args_dict["step_args"]["io"][
            "concept_encoder_model_key"
        ]
        self.output_key = args_dict["step_args"]["io"]["output_key"]

        self.mmmt_pipeline = args_dict["pipeline"]
        self.pipeline = args_dict["pipeline"]["fuse_pipeline"]

    def __call__(
        self,
    ):
        """
        Concatenates encoded modalities and trains concept AE.
        The trained concept enccoder model is stored in te pipeline object.

        """
        with mlflow.start_run(run_name=f"{self.__class__.__qualname__}", nested=True):
            mlflow.log_params(NDict(self.concept_encoding_strategy).flatten())
            (
                concept_training_dataset,
                concept_validation_dataset,
                concept_test_dataset,
            ) = self.concatenate_encoded_modalities()

            concept_encoder_model = self.encode_concepts(
                concept_training_dataset,
                concept_validation_dataset,
            )
            self.mmmt_pipeline[self.output_key] = {
                "concatenated_training_dataset": concept_training_dataset,
                "concatenated_validation_dataset": concept_validation_dataset,
                "concatenated_test_dataset": concept_test_dataset,
            }

            self.mmmt_pipeline[self.concept_encoder_model_key] = concept_encoder_model

    def concatenate_encoded_modalities(
        self,
    ):
        """
        Extends fuse pipline to concatenate the encoded modalities and executes the fuse pipeline to get the concatenated dataset for successive concept AE training.

        Returns:
            concatenated training and validation set for concept AE training
        """
        # Extend pipeline for representation
        concatenating_keys = self.concept_encoding_strategy["io"]["input_keys"]
        self.pipeline.extend(
            [
                (
                    OpConcat(),
                    dict(
                        keys_in=concatenating_keys,
                        key_out=self.output_key,
                        axis=0,
                    ),
                ),
            ]
        )

        if self.concept_encoding_strategy["add_feature_names"]:
            self.pipeline.extend(
                [
                    (
                        OpConcatNames(),
                        dict(
                            keys_in=concatenating_keys,
                            key_out="names." + self.output_key,
                        ),
                    ),
                ]
            )

        if self.cache_concept_config is None:
            self.cache_concept_config = {}
        if "cache_dirs" not in self.cache_concept_config:
            self.cache_concept_config["cache_dirs"] = [
                os.path.join(self.root_dir, "cache_concept")
            ]

        cacher_concept = SamplesCacher(
            "cache_concept",
            self.pipeline,
            **self.cache_concept_config,
        )
        concept_training_dataset = DatasetDefault(
            sample_ids=self.training_sample_ids,
            static_pipeline=self.pipeline,
            cacher=cacher_concept,
        )
        concept_training_dataset.create()

        concept_validation_dataset = DatasetDefault(
            sample_ids=self.val_sample_ids,
            static_pipeline=self.pipeline,
            cacher=cacher_concept,
        )
        concept_validation_dataset.create()

        concept_test_dataset = DatasetDefault(
            sample_ids=self.test_sample_ids,
            static_pipeline=self.pipeline,
            cacher=cacher_concept,
        )
        concept_test_dataset.create()

        logging.info("concatenated unimodal representation created")

        return (
            concept_training_dataset,
            concept_validation_dataset,
            concept_test_dataset,
        )

    def encode_concepts(
        self,
        concept_training_dataset,
        concept_validation_dataset,
    ):
        """
        Encoding concepts from concatenated representation

        Args:
            concept_training_dataset: training datasets containing the concatenated representation
            concept_validation_dataset: validation datasets containing the concatenated representation

        Returns:
            trained concept encoder
        """
        concept_train_dataloader = DataLoader(
            dataset=concept_training_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            num_workers=self.num_workers,
        )

        concept_validation_dataloader = DataLoader(
            dataset=concept_validation_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            num_workers=self.num_workers,
        )

        # Multimodal representation

        unimodal_features = len(concept_training_dataset[0][self.output_key])
        encoding_layers = self.concept_encoding_strategy.get("encoding_layers", [16])
        autoencoder = mmmt.data.representation.AutoEncoderBuilder(
            unimodal_features, encoding_layers
        )
        concept_encoder_model = autoencoder.encoder()
        concept_decoder_model = autoencoder.decoder()

        concept_representation = mmmt.data.representation.AutoEncoderTrainer(
            concept_encoder_model,
            self.output_key,
            "model.embedding.multimodal",
            concept_decoder_model,
            "model.recon.multimodal",
        )
        if self.concept_train_config is None:
            self.concept_train_config = {}
        if "model_dir" not in self.concept_train_config:
            self.concept_train_config["model_dir"] = os.path.join(
                self.root_dir, "model_dir_concept"
            )
        else:
            self.concept_train_config["model_dir"] = os.path.join(
                self.root_dir, self.concept_train_config["model_dir"]
            )

        concept_representation.set_train_config(**self.concept_train_config)
        ckpt_file = os.path.join(
            self.concept_train_config["model_dir"], "best_epoch.ckpt"
        )
        use_pretrained = self.concept_encoding_strategy.get("use_pretrained", False)

        if not use_pretrained or not os.path.exists(ckpt_file):
            logging.debug(f"Pretrained: {use_pretrained}")
            logging.debug(f"{ckpt_file} exists: {os.path.exists(ckpt_file)}")
            logging.info("Starting training of AE for concepts")
            concept_representation.fit(
                concept_train_dataloader, concept_validation_dataloader
            )
        else:
            logging.info(
                f"Training of concept AE is skipped as trained model already exists in {os.path.exists(ckpt_file)}"
            )

        concept_representation.load_checkpoint(ckpt_file)

        self.pipeline.extend(
            [
                (
                    OpForwardPass(concept_encoder_model, 1),
                    dict(
                        key_in=self.output_key,
                        key_out="data.forward_pass.multimodal",
                    ),
                ),
                (OpToNumpy(), dict(key="data.forward_pass.multimodal")),
            ]
        )

        return concept_encoder_model
