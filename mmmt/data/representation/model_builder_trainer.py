from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Union

import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses.loss_base import LossBase
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.eval.metrics.metrics_common import MetricBase
from torch.utils.data.dataloader import DataLoader


class ModelBuilderTrainer:
    """
    Class containing methods to train the model constructed using the model builder
    """

    def __init__(
        self,
        instantiated_model: torch.nn.Module,
        model_in_key: str,
        model_out_key: str,
        label_key: str,
    ) -> None:
        """
        Args:
            instantiated_model: configured model to be trained
            model_in_key: key in the FuseMedML pipeline where the samples to train the model (e.g. derived graphs) are stored
            model_out_key: key in the FuseMedML pipeline where the model output should be stored
            label_key: key in the FuseMedML pipeline where the labels are stored
        """
        # store arguments
        self._instantiated_model = instantiated_model
        self._model_in_key = model_in_key
        self._model_out_key = model_out_key
        self._label_key = label_key

        # set to None train configuration - should be set
        self._model_dir = None
        self._losses = None
        self._best_epoch_source = None
        self._optimizers_and_lr_schs = None
        self._train_metrics = None
        self._validation_metrics = None
        self._callbacks = None

        # wrap the model
        self._model = ModelWrapSeqToDict(
            model=self._instantiated_model,
            model_inputs=[self._model_in_key],
            model_outputs=[self._model_out_key],
        )

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

        if losses is None:
            losses = {
                "cls_loss": LossDefault(
                    pred=self._model_out_key,
                    target=self._label_key,
                    callable=F.cross_entropy,
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
        )

        # train
        mlflow.pytorch.autolog()
        pl_trainer.fit(pl_module, train_dataloader, validation_dataloader)

    def load_checkpoint(self, checkpoint_filename: str) -> torch.nn.Module:
        LightningModuleDefault.load_from_checkpoint(
            checkpoint_filename,
            model_dir=self._model_dir,
            model=self._model,
            map_location="cpu",
            strict=True,
        )
        return self._model

    def predict(
        self,
        infer_dataloader: DataLoader,
        model_dir: str,
        checkpoint_filename: str,
        keys_to_extract: Sequence[str],
    ) -> pd.DataFrame:
        """
        Method for using the model to predict the label of samples

        Args:
            infer_dataloader: dataloader to infer - each batch expected to be a dictionary batch_dict)
            model_dir: path to directory with checkpoints
            checkpoint_filename: path to checkpoint file as stored in self.fit() method
            keys_to_extract: sequence of keys to extract and dump into the dataframe

        Returns:
            dataframe containing the predictions
        """

        pl_module = LightningModuleDefault.load_from_checkpoint(
            checkpoint_filename,
            model_dir=model_dir,
            model=self._model,
            map_location="cpu",
            strict=True,
        )
        pl_module.set_predictions_keys(keys_to_extract)

        # create lightning trainer.
        pl_trainer = pl.Trainer(
            default_root_dir=self._model_dir,
            accelerator=self._pl_trainer_accelerator,
            strategy=self._pl_trainer_strategy,
            devices=self._pl_trainer_devices,
            auto_select_gpus=True,
        )

        preds = pl_trainer.predict(
            model=pl_module, dataloaders=infer_dataloader, return_predictions=True
        )

        df = convert_predictions_to_dataframe(preds)

        return df
