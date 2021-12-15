import math
from typing import Tuple, Optional

from util import matplotlib_imshow

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import Tensor
from pytorch_lightning.callbacks import Callback
from sklearn.preprocessing import LabelEncoder


class LogWorstPredictions(Callback):
    """
    At the end of every epoch, log the predictions with the highest loss values,
    i.e. the worst predictions of the model.
    """

    def __init__(self):
        pass

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        pass
        # TODO


class LogModelPredictionsMAML(Callback):
    """
    Use a fixed test batch to monitor model predictions at the end of every epoch.

    Specifically: it generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's prediction alongside the actual target.
    """

    def __init__(
        self,
        label_encoder: "LabelEncoder",
        val_batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        train_batch: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        use_gpu: bool = True,
        data_format: str = "word",
        enable_grad: bool = False,
        predict_on_train_start: bool = False,
    ):
        self.label_encoder = label_encoder
        self.val_batch = val_batch
        self.train_batch = train_batch
        self.use_gpu = use_gpu
        self.data_format = data_format
        self.enable_grad = enable_grad
        self.predict_on_train_start = predict_on_train_start

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        self._predict_intermediate(trainer, pl_module, split="val")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.train_batch is not None:
            self._predict_intermediate(trainer, pl_module, split="train")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.predict_on_train_start:
            self._predict_intermediate(trainer, pl_module, split="val")
            if self.train_batch is not None:
                self._predict_intermediate(trainer, pl_module, split="train")

        # Log the support images once at the start of training.
        imgs, targets, *_ = self.val_batch
        self._log_intermediate(
            trainer, pl_module, imgs, targets, split="val", plot_title="support batch"
        )
        if self.train_batch is not None:
            imgs, targets, *_ = self.train_batch
            self._log_intermediate(
                trainer,
                pl_module,
                imgs,
                targets,
                split="train",
                plot_title="support batch",
            )

    def _predict_intermediate(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", split="val"
    ):
        """Make predictions on a fixed batch of data; log the results to Tensorboard."""

        if split == "train":
            batch = self.train_batch
        else:  # split == "val"
            batch = self.val_batch

        # Make predictions.
        support_imgs, support_tgts, query_imgs, query_tgts = batch
        pl_module.eval()
        torch.set_grad_enabled(self.enable_grad)
        _, preds, *_ = pl_module(
            *[
                t.cuda() if self.use_gpu else t
                for t in [support_imgs, support_tgts, query_imgs]
            ]
        )
        torch.set_grad_enabled(False)

        # Log the results.
        imgs, targets, *_ = batch
        self._log_intermediate(
            trainer, pl_module, query_imgs, query_tgts, preds, split=split
        )

    def _log_intermediate(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        imgs: Tensor,
        targets: Tensor,
        preds: Optional[Tensor] = None,
        split: str = "val",
        plot_title: str = "query predictions vs targets",
    ):
        """Log a batch of images along with their targets to Tensorboard."""

        # Find padding and <EOS> positions in predictions and targets.
        eos_idxs_pred = None
        if preds is not None:
            eos_idxs_pred = (
                (preds == pl_module.decoder.eos_tkn_idx).float().argmax(1).tolist()
            )
        eos_idxs_tgt = (
            (targets == pl_module.decoder.eos_tkn_idx).float().argmax(1).tolist()
        )

        # Generate plot.
        fig = plt.figure(figsize=(12, 16))
        for i, t in enumerate(targets.tolist()):
            # Decode predictions and targets.
            p = None
            if preds is not None:
                p = preds.tolist()[i]
            max_target_idx = eos_idxs_tgt[i]
            pred_str = None
            if eos_idxs_pred:
                max_pred_idx = eos_idxs_pred[i]
                p = p[1:]  # skip the initial <SOS> token, which is added by default
                if max_pred_idx != 0:
                    pred_str = "".join(
                        self.label_encoder.inverse_transform(p)[:max_pred_idx]
                    )
                else:
                    pred_str = "".join(self.label_encoder.inverse_transform(p))
            if max_target_idx != 0:
                target_str = "".join(
                    self.label_encoder.inverse_transform(t)[:max_target_idx]
                )
            else:
                target_str = "".join(self.label_encoder.inverse_transform(t))

            # Create plot.
            ncols = 2 if self.data_format == "word" else 1
            nrows = math.ceil(targets.size(0) / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(imgs[i])
            ttl = f"Target: {target_str}"
            if pred_str is not None:
                ttl = f"Pred: {pred_str}\n" + ttl
            ax.set_title(ttl)

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(f"{split}: {plot_title}", fig, trainer.global_step)
