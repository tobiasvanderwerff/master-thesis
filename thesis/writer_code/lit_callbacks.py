from typing import Tuple, Optional

from thesis.lit_callbacks import (
    LogModelPredictionsCallback,
    LogWorstPredictionsCallback,
)
from thesis.writer_code.util import AVAILABLE_MODELS
from thesis.util import TrainMode

from htr.util import LabelEncoder

import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader


class LogWorstPredictions(LogWorstPredictionsCallback):
    def __init__(
        self,
        label_encoder: LabelEncoder,
        shots: int,
        ways: int,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        training_skipped: bool = False,
    ):
        super().__init__(
            label_encoder=label_encoder,
            shots=shots,
            ways=ways,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            training_skipped=training_skipped,
        )

    def log_worst_predictions(
        self,
        dataloader: DataLoader,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        mode: TrainMode = TrainMode.TRAIN,
        forward_mode: Optional[str] = "val",
    ):
        super().log_worst_predictions(
            dataloader, trainer, pl_module, mode, forward_mode=forward_mode
        )

    def _load_best_model(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        label_encoder: LabelEncoder,
    ):
        (
            ckpt_callback,
            best_model_path,
            model_hparams_file,
            args,
        ) = super().get_args_for_loading_model(trainer, pl_module, label_encoder)
        args.update(
            dict(
                num_writers=pl_module.num_writers,
                writer_emb_type=pl_module.writer_emb_type.name,
                code_size=pl_module.code_size,
                adaptation_num_hidden=pl_module.adaptation_num_hidden,
                ways=pl_module.ways,
                shots=pl_module.shots,
                learning_rate_emb=pl_module.learning_rate_emb,
            )
        )

        cls = pl_module.__class__
        cls_name = cls.__name__
        assert cls_name in AVAILABLE_MODELS, (
            f"Unrecognized model class: {cls}. " f"Choices: {AVAILABLE_MODELS}"
        )
        model = cls.init_with_base_model_from_checkpoint(**args)
        trainer.model = model


class LogModelPredictions(LogModelPredictionsCallback):
    def __init__(
        self,
        label_encoder: LabelEncoder,
        val_batch: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, str]] = None,
        train_batch: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, str]] = None,
        data_format: str = "word",
        predict_on_train_start: bool = False,
    ):
        super().__init__(
            label_encoder=label_encoder,
            val_batch=val_batch,
            train_batch=train_batch,
            data_format=data_format,
            predict_on_train_start=predict_on_train_start,
        )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.predict_on_train_start:
            if self.val_batch is not None:
                self._predict_intermediate(trainer, pl_module, split=TrainMode.VAL)
            if self.train_batch is not None:
                self._predict_intermediate(trainer, pl_module, split=TrainMode.TRAIN)

        # Log the support images once at the start of training.
        if self.val_batch is not None:
            support_imgs, support_targets, _, _ = self.val_batch
            self._log_intermediate(
                trainer,
                support_imgs,
                support_targets,
                split=TrainMode.VAL,
                plot_title="support batch",
            )
        if self.train_batch is not None:
            support_imgs, support_targets, _ = self.train_batch
            self._log_intermediate(
                trainer,
                support_imgs,
                support_targets,
                split=TrainMode.TRAIN,
                plot_title="support batch",
            )

    def _predict_intermediate(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        split: TrainMode = TrainMode.VAL,
    ):
        """Make predictions on a fixed batch of data; log the results to Tensorboard."""

        device = "cuda:0" if pl_module.on_gpu else "cpu"
        if split == TrainMode.TRAIN:
            batch = self.train_batch
        else:  # split == "val"
            batch = self.val_batch

        # Make predictions.
        if split == TrainMode.TRAIN:
            query_imgs, query_tgts, writer_ids = batch
            _, preds, *_ = pl_module.model(
                *[t.to(device) for t in [query_imgs, query_tgts, writer_ids]],
                mode="train",
            )
        else:  # val/test
            support_imgs, support_tgts, query_imgs, query_tgts = batch
            torch.set_grad_enabled(True)
            _, preds, *_ = pl_module(*[t.to(device) for t in batch], mode="val")
            torch.set_grad_enabled(False)

        # Log the results.
        self._log_intermediate(
            trainer,
            query_imgs,
            query_tgts,
            preds,
            split=split,
        )
