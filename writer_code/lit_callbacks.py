import math
from typing import Tuple, Optional
from pathlib import Path

from metahtr.util import decode_prediction

from htr.data import IAMDataset
from htr.util import matplotlib_imshow, LabelEncoder
from htr.models.sar.sar import ShowAttendRead
from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder

from lit_models import WriterCodeAdaptiveModel

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


PREDICTIONS_TO_LOG = {
    "word": 10,
    "line": 6,
    "form": 1,
}


class LogWorstPredictions(Callback):
    """
    At the end of training, log the worst image prediction, meaning the predictions
    with the highest character error rates.
    """

    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        training_skipped: bool = False,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.training_skipped = training_skipped

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.training_skipped and self.val_dataloader is not None:
            self.log_worst_predictions(
                self.val_dataloader, trainer, pl_module, mode="val"
            )

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.test_dataloader is not None:
            self.log_worst_predictions(
                self.test_dataloader, trainer, pl_module, mode="test"
            )

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.train_dataloader is not None:
            self.log_worst_predictions(
                self.train_dataloader, trainer, pl_module, mode="train"
            )
        if self.val_dataloader is not None:
            self.log_worst_predictions(
                self.val_dataloader, trainer, pl_module, mode="val"
            )

    def log_worst_predictions(
        self,
        dataloader: DataLoader,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        mode: str = "train",
    ):
        img_cers = []
        device = "cuda:0" if pl_module.on_gpu else "cpu"
        if not self.training_skipped:
            self._load_best_model(trainer, pl_module)
            pl_module = trainer.model

        print(f"Running {mode} inference on best model...")

        eos_tkn_idx = pl_module.model.eos_tkn_idx
        # Run inference on the validation set.
        torch.set_grad_enabled(True)
        shots, ways = pl_module.shots, pl_module.ways
        for imgs, targets, writer_ids in dataloader:
            for task in range(ways):
                writer_ids_uniq = writer_ids.unique().tolist()
                task_slice = writer_ids == writer_ids_uniq[task]
                tsk_imgs, tsk_tgts = imgs[task_slice], targets[task_slice]
                support_imgs, support_tgts, query_imgs, query_tgts = (
                    tsk_imgs[:shots],
                    tsk_tgts[:shots],
                    tsk_imgs[shots:],
                    tsk_tgts[shots:],
                )

                _, preds, *_ = pl_module(
                    *[t.to(device) for t in [support_imgs, support_tgts, query_imgs]]
                )

                cer_metric = pl_module.model.cer_metric
                for prd, tgt, im in zip(preds, query_tgts, query_imgs):
                    with torch.inference_mode():
                        cer_metric.reset()
                        cer = cer_metric(prd.unsqueeze(0), tgt.unsqueeze(0)).item()
                        img_cers.append((im, cer, prd, tgt))
        torch.set_grad_enabled(False)

        # Log the worst k predictions.
        to_log = PREDICTIONS_TO_LOG["word"] * 2
        img_cers.sort(key=lambda x: x[1], reverse=True)  # sort by CER
        img_cers = img_cers[:to_log]
        fig = plt.figure(figsize=(24, 16))
        for i, (im, cer, prd, tgt) in enumerate(img_cers):
            pred_str = decode_prediction(
                prd[1:],
                pl_module.model.label_encoder,
                eos_tkn_idx,
            )
            target_str = decode_prediction(
                tgt, pl_module.model.label_encoder, eos_tkn_idx
            )

            # Create plot.
            ncols = 4
            nrows = math.ceil(to_log / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(im, IAMDataset.MEAN, IAMDataset.STD)
            ax.set_title(f"Pred: {pred_str} (CER: {cer:.2f})\nTarget: {target_str}")

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(f"{mode}: worst predictions", fig, trainer.global_step)
        plt.close(fig)

        print("Done.")

    @staticmethod
    def _load_best_model(trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        ckpt_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_callback = cb
                break
        assert ckpt_callback is not None, "ModelCheckpoint not found in callbacks."
        best_model_path = ckpt_callback.best_model_path

        print(f"Loading best model at {best_model_path}")

        model_hparams_file = Path(best_model_path).parent.parent / "model_hparams.yaml"
        args = dict(
            checkpoint_path=best_model_path,
            model_hparams_file=model_hparams_file,
            label_encoder=pl_module.model.label_encoder,
            load_meta_weights=True,
            taskset_train=pl_module.taskset_train,
            taskset_val=pl_module.taskset_val,
            taskset_test=pl_module.taskset_test,
            ways=pl_module.ways,
            shots=pl_module.shots,
            num_workers=pl_module.num_workers,
        )

        if isinstance(pl_module.model, FullPageHTREncoderDecoder):
            model = WriterCodeAdaptiveModel.init_with_base_model_from_checkpoint(
                "fphtr", **args
            )
        elif isinstance(pl_module.model, ShowAttendRead):
            model = WriterCodeAdaptiveModel.init_with_base_model_from_checkpoint(
                "sar", **args
            )
        else:
            raise ValueError(f"Unrecognized model class: {pl_module.model.__class__}")

        trainer.model = model


class LogModelPredictions(Callback):
    """
    Use a fixed test batch to monitor model predictions at the end of every epoch.

    Specifically: it generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's prediction alongside the actual target.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        val_batch: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        train_batch: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        data_format: str = "word",
        predict_on_train_start: bool = False,
    ):
        self.label_encoder = label_encoder
        self.val_batch = val_batch
        self.train_batch = train_batch
        self.data_format = data_format
        self.predict_on_train_start = predict_on_train_start

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.val_batch is not None:
            self._predict_intermediate(trainer, pl_module, split="val")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.train_batch is not None:
            self._predict_intermediate(trainer, pl_module, split="train")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.predict_on_train_start:
            if self.val_batch is not None:
                self._predict_intermediate(trainer, pl_module, split="val")
            if self.train_batch is not None:
                self._predict_intermediate(trainer, pl_module, split="train")

        # Log the support images once at the start of training.
        if self.val_batch is not None:
            support_imgs, support_targets, _, _ = self.val_batch
            self._log_intermediate(
                trainer,
                pl_module,
                support_imgs,
                support_targets,
                split="val",
                plot_title="support batch",
            )
        if self.train_batch is not None:
            support_imgs, support_targets, _ = self.train_batch
            self._log_intermediate(
                trainer,
                pl_module,
                support_imgs,
                support_targets,
                split="train",
                plot_title="support batch",
            )

    def _predict_intermediate(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", split="val"
    ):
        """Make predictions on a fixed batch of data; log the results to Tensorboard."""

        device = "cuda:0" if pl_module.on_gpu else "cpu"
        if split == "train":
            batch = self.train_batch
        else:  # split == "val"
            batch = self.val_batch

        # Make predictions.
        if split == "train":
            query_imgs, query_tgts, writer_ids = batch
            wrtr_emb = pl_module.writer_embs(writer_ids.to(device))  # (N, emb_size)
            inp = (query_imgs.to(device), wrtr_emb, query_tgts.to(device))
            logits, loss = pl_module.base_model_forward(*inp, teacher_forcing=True)
            preds = logits.argmax(-1)
        else:  # val/test
            support_imgs, support_tgts, query_imgs, query_tgts = batch
            torch.set_grad_enabled(True)
            _, preds, *_ = pl_module(*[t.to(device) for t in batch])
            torch.set_grad_enabled(False)

        # Log the results.
        self._log_intermediate(
            trainer,
            pl_module,
            query_imgs,
            query_tgts,
            preds,
            split=split,
        )

    def _log_intermediate(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        imgs: Tensor,
        targets: Tensor,
        preds: Optional[Tensor] = None,
        split: str = "val",
        plot_suptitle: Optional[str] = None,
        plot_title: str = "query predictions vs targets",
    ):
        """Log a batch of images along with their targets to Tensorboard."""

        assert imgs.shape[0] == targets.shape[0]

        eos_tkn_idx = pl_module.model.eos_tkn_idx
        # Generate plot.
        fig = plt.figure(figsize=(12, 16))
        for i, (tgt, im) in enumerate(zip(targets, imgs)):
            # Decode predictions and targets.
            pred_str = None
            if preds is not None:
                pred_str = decode_prediction(preds[i], self.label_encoder, eos_tkn_idx)
            target_str = decode_prediction(tgt, self.label_encoder, eos_tkn_idx)

            # Create plot.
            ncols = 2 if self.data_format == "word" else 1
            nrows = math.ceil(targets.size(0) / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(im, IAMDataset.MEAN, IAMDataset.STD)
            ttl = f"Target: {target_str}"
            if pred_str is not None:
                ttl = f"Pred: {pred_str}\n" + ttl
            ax.set_title(ttl)
        if plot_suptitle is not None:
            fig.suptitle(plot_suptitle)

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(f"{split}: {plot_title}", fig, trainer.global_step)
        plt.close(fig)
