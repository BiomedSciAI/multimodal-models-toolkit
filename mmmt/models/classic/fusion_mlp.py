from copy import deepcopy
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader

from fuse.dl.models.heads.head_1D_classifier import Head1DClassifier
from fuse.dl.losses import LossBase, LossDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.eval import MetricBase
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC


class FusionMLPClassifer:
    """
    Basic feature (mid) fusion algorithm using MLP, including training.
    """

    def __init__(
        self, input_keys: Sequence[Tuple[str, int]], target_key: str, **arch_kwargs
    ):
        """
        :param input_keys:  List of feature map inputs - tuples of (batch_dict key, channel depth)
                            If multiple inputs are used, they are concatenated on the channel axis
                            for example:
                                input_keys=(('model.backbone_features', 193),)
        :param arch_kwargs: arguments to create the MLP - see Head1DClassifier
        :param target_key: labels key
        """

        self._input_keys = input_keys
        self._arch_kwargs = arch_kwargs
        self._scores_key = "model.output.classifier"
        self._logits_key = "model.logits.classifier"
        self._target_key = target_key

        self._model = Head1DClassifier("classifier", input_keys, **arch_kwargs)

        # set to None train configuration - should be set
        self._model_dir = None
        self._losses = None
        self._best_epoch_source = None
        self._optimizers_and_lr_schs = None
        self._train_metrics = None
        self._validation_metrics = None
        self._callbacks = None

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
        self._model_dir = model_dir
        if losses is None:
            self._losses = {
                "cls_loss": LossDefault(
                    pred=self._logits_key,
                    target=self._target_key,
                    callable=F.cross_entropy,
                    weight=1.0,
                ),
            }
        else:
            self._losses = losses

        if train_metrics is None:
            self._train_metrics = {
                "auc": MetricAUCROC(pred=self._scores_key, target=self._target_key)
            }
        else:
            self._train_metrics = train_metrics

        if validation_metrics is None:
            self._validation_metrics = deepcopy(
                self._train_metrics
            )  # use the same metrics in validation as well
        else:
            self._validation_metrics = validation_metrics

        # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
        if best_epoch_source is None:
            # assumes binary classification and that MetricAUCROC is in validation_metrics
            self._best_epoch_source = dict(
                monitor="validation.metrics.auc",
                mode="max",
            )
        else:
            self._best_epoch_source

        if optimizers_and_lr_schs is None:
            optimizer = optim.SGD(
                self._model.parameters(),
                lr=1e-3,
                weight_decay=0.0,
                momentum=0.9,
                nesterov=True,
            )

        # create learning scheduler
        if optimizers_and_lr_schs is None:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            lr_sch_config = dict(
                scheduler=lr_scheduler, monitor="validation.losses.total_loss"
            )

            # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
            self._optimizers_and_lr_schs = dict(
                optimizer=optimizer, lr_scheduler=lr_sch_config
            )
        else:
            self._optimizers_and_lr_schs = optimizers_and_lr_schs

        self._callbacks = callbacks
        self._pl_trainer_num_epochs = pl_trainer_num_epochs
        self._pl_trainer_accelerator = pl_trainer_accelerator
        self._pl_trainer_devices = pl_trainer_devices
        self._pl_trainer_strategy = pl_trainer_strategy

    def fit(self, train_dataloader: DataLoader, validation_dataloader: DataLoader):
        pl_module = LightningModuleDefault(
            model_dir=self._model_dir,
            model=self._model,
            losses=self._losses,
            train_metrics=self._train_metrics,
            validation_metrics=self._validation_metrics,
            best_epoch_source=self._best_epoch_source,
            optimizers_and_lr_schs=self._optimizers_and_lr_schs,
        )

        # create lightining trainer.
        pl_trainer = pl.Trainer(
            default_root_dir=self._model_dir,
            max_epochs=self._pl_trainer_num_epochs,
            accelerator=self._pl_trainer_accelerator,
            devices=self._pl_trainer_devices,
            auto_select_gpus=True,
        )

        # train
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

    def model(self) -> torch.nn.Module:
        return self._model
