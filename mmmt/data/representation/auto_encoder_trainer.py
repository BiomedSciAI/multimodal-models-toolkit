import copy
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.optim as optim
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses.loss_base import LossBase
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.eval.metrics.metrics_common import MetricBase
from torch.utils.data.dataloader import DataLoader


class AutoEncoderBuilder:
    """Builds an encoder and a decoder that are symmetrical"""

    def __init__(self, input_features, encoding_dimensions) -> None:
        """Contructor method

        :param input_features: number of input features
        :type input_features: int
        :param encoding_dimensions: number of neurons per encoding layer
        :type encoding_dimensions: (list)
        """
        self.original_features = input_features
        self.encoding_dimensions = encoding_dimensions

    def encoder(self):
        """instantiates the encoder model

        :return: encoder model
        :rtype: torch.nn.Module
        """
        model = torch.nn.Sequential()
        in_features = self.original_features
        for out_features in self.encoding_dimensions:
            model.append(torch.nn.Linear(in_features, out_features))
            model.append(torch.nn.ReLU(True))
            in_features = out_features

        return model

    def decoder(self):
        """instantiates the decoder model

        :return: decoder model
        :rtype: torch.nn.Module
        """
        model = torch.nn.Sequential()
        decoding_dimensions = copy.deepcopy(self.encoding_dimensions)
        in_features = decoding_dimensions.pop()
        decoding_dimensions.reverse()
        for out_features in decoding_dimensions:
            model.append(torch.nn.Linear(in_features, out_features))
            model.append(torch.nn.ReLU(True))
            in_features = out_features
        model.append(torch.nn.Linear(in_features, self.original_features))
        return model


class AutoEncoderTrainer:
    """Trains an autoencoder using FuseMedML"""

    def __init__(
        self,
        encoder: torch.nn.Module,
        encoder_in_key: str,
        embedding_key: str,
        decoder: torch.nn.Module,
        decoder_out_key: str,
    ) -> None:
        """_summary_

        :param encoder: encoder model
        :type encoder: torch.nn.Module
        :param encoder_in_key: decoder model
        :type encoder_in_key: str
        :param embedding_key: name of the the key that will store the embedding in the sample_dict
        :type embedding_key: str
        :param decoder: decoder model
        :type decoder: torch.nn.Module
        :param decoder_out_key: name of the key that will store the reconstruction in the sample_dict
        :type decoder_out_key: str
        """
        # store arguments
        self._encoder = encoder
        self._encoder_in_key = encoder_in_key
        self._embedding_key = embedding_key
        self._decoder = decoder
        self._decoder_out_key = decoder_out_key

        # set to None train configuration - should be set
        self._model_dir = None
        self._losses = None
        self._best_epoch_source = None
        self._optimizers_and_lr_schs = None
        self._train_metrics = None
        self._validation_metrics = None
        self._callbacks = None

        # wrap the encoder and decoder
        self._fuse_encoder = ModelWrapSeqToDict(
            model=self._encoder,
            model_inputs=[self._encoder_in_key],
            model_outputs=[self._embedding_key],
        )
        self._fuse_decoder = ModelWrapSeqToDict(
            model=self._decoder,
            model_inputs=[self._embedding_key],
            model_outputs=[self._decoder_out_key],
        )
        self._model = torch.nn.Sequential(self._fuse_encoder, self._fuse_decoder)

    def set_train_config(
        self,
        model_dir: str,
        losses: Optional[Dict[str, LossBase]] = None,
        best_epoch_source: Optional[Union[Dict, List[Dict]]] = None,
        optimizers_and_lr_schs: Any = None,
        train_metrics: Optional[OrderedDict[str, MetricBase]] = None,
        validation_metrics: Optional[OrderedDict[str, MetricBase]] = None,
        callbacks: Optional[Sequence[pl.Callback]] = None,
        pl_trainer_num_epochs: int = 100,
        pl_trainer_accelerator: str = "gpu",
        pl_trainer_devices: int = 1,
        pl_trainer_strategy: Optional[str] = None,
    ):
        """_summary_

        :param model_dir: folder name to store the checkpoints of the model
        :type model_dir: str
        :param losses: losses to use during training, defaults to None
        :type losses: Optional[Dict[str, LossBase]], optional
        :param best_epoch_source: criteria for choosing best epoch as defined by PyTorch Lightning, defaults to None
        :type best_epoch_source: Optional[Union[Dict, List[Dict]]], optional
        :param optimizers_and_lr_schs: optimizers and learning rate schedulers as defined by PyTorch Lightning, defaults to None
        :type optimizers_and_lr_schs: Any, optional
        :param train_metrics: metrics to evaluate during training, defaults to None
        :type train_metrics: Optional[OrderedDict[str, MetricBase]], optional
        :param validation_metrics: metrics to evaluate during validation, defaults to None
        :type validation_metrics: Optional[OrderedDict[str, MetricBase]], optional
        :param callbacks: additional PyTorch Lightning callbacks, defaults to None
        :type callbacks: Optional[Sequence[pl.Callback]], optional
        :param pl_trainer_num_epochs: number of epochs to train, defaults to 100
        :type pl_trainer_num_epochs: int, optional
        :param pl_trainer_accelerator: whether to use "cpu" or "gpu", defaults to "gpu"
        :type pl_trainer_accelerator: str, optional
        :param pl_trainer_devices: number of devices used by PyTorch Lightning, defaults to 1
        :type pl_trainer_devices: int, optional
        :param pl_trainer_strategy: PyTorch Lightning trainer strategy, defaults to None
        :type pl_trainer_strategy: Optional[str], optional
        """

        if losses is None:
            losses = {
                "reconst": LossDefault(
                    pred=self._decoder_out_key,
                    target=self._encoder_in_key,
                    callable=torch.nn.MSELoss(),
                    weight=1.0,
                ),
            }

        if best_epoch_source is None:
            best_epoch_source = dict(monitor="validation.losses.total_loss", mode="min")

        if optimizers_and_lr_schs is None:
            # create optimizer
            optimizer = optim.Adam(
                self._model.parameters(),
                lr=1e-4,
                weight_decay=0.001,
            )

            # create learning scheduler
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            lr_sch_config = dict(
                scheduler=lr_scheduler, monitor="validation.losses.total_loss"
            )

            # optimizer and lr sch - see pl.LightningModule.configure_optimizers return value for all options
            optimizers_and_lr_schs = dict(
                optimizer=optimizer, lr_scheduler=lr_sch_config
            )

        self._model_dir = model_dir
        self._losses = losses
        self._best_epoch_source = best_epoch_source
        self._optimizers_and_lr_schs = optimizers_and_lr_schs

        self._train_metrics = train_metrics
        self._validation_metrics = validation_metrics
        self._callbacks = callbacks

        self._pl_trainer_num_epochs = pl_trainer_num_epochs
        self._pl_trainer_accelerator = pl_trainer_accelerator
        self._pl_trainer_devices = pl_trainer_devices
        self._pl_trainer_strategy = pl_trainer_strategy

    def fit(
        self, train_dataloader: DataLoader, validation_dataloader: DataLoader
    ) -> None:
        """trains the model

        :param train_dataloader: dataloader for training
        :type train_dataloader: DataLoader
        :param validation_dataloader: dataloader for validation
        :type validation_dataloader: DataLoader
        """
        assert (
            self._model_dir is not None
        ), "Error expecting train configuration. Call to method set_train_config() to set it"

        pl_module = LightningModuleDefault(
            model_dir=self._model_dir,
            model=self._model,
            losses=self._losses,
            train_metrics=self._train_metrics,
            validation_metrics=self._validation_metrics,
            best_epoch_source=self._best_epoch_source,
            optimizers_and_lr_schs=self._optimizers_and_lr_schs,
        )

        # create lightning trainer.
        pl_trainer = pl.Trainer(
            default_root_dir=self._model_dir,
            max_epochs=self._pl_trainer_num_epochs,
            accelerator=self._pl_trainer_accelerator,
            strategy=self._pl_trainer_strategy,
            devices=self._pl_trainer_devices,
            auto_select_gpus=True,
            num_sanity_val_steps=0,
        )

        # train

        mlflow.pytorch.autolog()
        pl_trainer.fit(pl_module, train_dataloader, validation_dataloader)

    def load_checkpoint(
        self, checkpoint_filename: str
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """loads a checkpoint

        :param checkpoint_filename: filename
        :type checkpoint_filename: str
        :return: encoder and decoder models
        :rtype: Tuple[torch.nn.Module, torch.nn.Module]
        """
        LightningModuleDefault.load_from_checkpoint(
            checkpoint_filename,
            model_dir=self._model_dir,
            model=self._model,
            map_location="cpu",
            strict=True,
        )
        return self._encoder, self._decoder
