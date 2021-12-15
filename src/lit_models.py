from typing import Optional, Dict, Union, Tuple
from pathlib import Path

from models import FullPageHTREncoderDecoder
from util import identity_collate_fn

import torch
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

import learn2learn as l2l


class MetaHTR(pl.LightningModule):
    def __init__(
        self,
        model: FullPageHTREncoderDecoder,
        taskset_train: Union[l2l.data.TaskDataset, Dataset],
        taskset_val: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_test: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        ways: int = 8,
        shots: int = 16,
        inner_lr: float = 0.0001,
        outer_lr: float = 0.0001,
        num_inner_steps: int = 1,
        num_workers: int = 0,
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        assert num_inner_steps >= 1

        self.taskset_train = taskset_train
        self.taskset_val = taskset_val
        self.taskset_test = taskset_test
        self.ways = ways
        self.shots = shots
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.num_workers = num_workers

        self.maml = l2l.algorithms.MAML(
            model, inner_lr, first_order=False, allow_nograd=True
        )

        self.save_hyperparameters(
            "ways",
            "shots",
            "inner_lr",
            "outer_lr",
            "num_inner_steps",
        )
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)

    def meta_learn(self, batch, mode="train") -> Tensor:
        outer_loss = 0.0

        imgs, target, writer_ids = batch
        writer_ids_uniq = writer_ids.unique().tolist()

        assert imgs.size(0) >= 2 * self.ways * self.shots, imgs.size(0)
        assert len(writer_ids.unique().tolist()) == self.ways

        # Split the batch into N different writers, where N = ways.
        for task in range(self.ways):  # tasks correspond to different writers
            wrtr_id = writer_ids_uniq[task]
            task_slice = writer_ids == wrtr_id
            imgs_, target_ = imgs[task_slice], target[task_slice]

            # Separate data into support/query set.
            support_indices = np.zeros(imgs_.size(0), dtype=bool)
            # Select first k even indices for support set.
            support_indices[np.arange(self.shots) * 2] = True
            query_indices = torch.from_numpy(~support_indices)
            support_indices = torch.from_numpy(support_indices)
            support_imgs, support_tgts = (
                imgs_[support_indices],
                target_[support_indices],
            )
            query_imgs, query_tgts = imgs_[query_indices], target_[query_indices]

            # Calling `maml.clone()` allows updating the module while still allowing
            # computation of derivatives of the new modules' parameters w.r.t. the
            # original parameters.
            learner = self.maml.clone()
            learner.train()

            # Inner loop.
            for _ in range(self.num_inner_steps):
                _, support_loss = learner.module.forward_teacher_forcing(
                    support_imgs, support_tgts
                )
                # Calculate gradients and take an optimization step.
                learner.adapt(support_loss)

            # Outer loop.
            # To me it is not fully clear whether to set the model to eval() for
            # the outer loop. The primary change is in the deactivation of dropout
            # layers and the usage of running activation statistics for
            # normalization instead of per-batch statistics.
            #
            # It seems logical that the outer loop activations are not recorded as
            # part of the running statistics for norm layers (since the adapted
            # parameters are not the same parameters obtained after doing a
            # gradient step). Deactivating dropout also seems reasonable since the
            # outer loop can be seen as a kind of evaluation of the updated model
            # from the inner loop. Therefore I choose to use model.eval() in the
            # outer loop.
            # learner.eval()
            if mode == "train":
                _, query_loss = learner.module.forward_teacher_forcing(
                    query_imgs, query_tgts
                )
            else:  # val/test
                with torch.inference_mode():
                    _, preds, query_loss = learner(query_imgs, query_tgts)

                # Log metrics.
                metrics = learner.module.calculate_metrics(preds, query_tgts)
                for metric, val in metrics.items():
                    self.log(metric, val, prog_bar=True)
            outer_loss += (1 / self.ways) * query_loss

        return outer_loss

    @property
    def encoder(self):
        return self.maml.module.encoder

    @property
    def decoder(self):
        return self.maml.module.decoder

    def training_step(self, batch, batch_idx):
        loss = self.meta_learn(batch, mode="train")
        self.log("train_loss", loss, sync_dist=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation requires finetuning a model in the inner loop, hence we need to
        # enable gradients.
        torch.set_grad_enabled(True)
        loss = self.meta_learn(batch, mode="val")
        torch.set_grad_enabled(False)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def forward(
        self,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
        inference_imgs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Do meta learning on a set of images and run inference on another set.

        Args:
            adaptation_imgs (Tensor): images to do adaptation on
            adaptation_targets (Tensor): targets for `adaptation_imgs`
            inference_imgs (Tensor): images to make predictions on
        Returns:
            predictions on `inference_imgs`, in the form of a 2-tuple:
                - logits, obtained at each time step during decoding
                - sampled class indices, i.e. model predictions, obtained by applying
                      greedy decoding (argmax on logits) at each time step
        """
        learner = self.maml.clone()
        learner.train()

        # Adapt the model.
        # For some reason using an autograd context manager like torch.enable_grad()
        # here does not work, perhaps due to some unexpected interaction between
        # Pytorch Lightning and the learn2learn lib. Therefore gradient
        # calculation should be set beforehand, outside of the current function.
        _, adaptation_loss = learner.module.forward_teacher_forcing(
            adaptation_imgs, adaptation_targets
        )
        learner.adapt(adaptation_loss)

        # Run inference on the adapted model.
        # learner.eval()
        with torch.inference_mode():
            logits, sampled_ids, _ = learner(inference_imgs)

        return logits, sampled_ids

    def train_dataloader(self):
        # Since we are using a l2l TaskDataset which already batches the data,
        # using a PyTorch DataLoader is redundant. However, Pytorch Lightning
        # requires the use of a proper DataLoader. Therefore, we pass a DataLoader
        # that acts as an identity function, simply passing a single batch of data
        # prepared by the TaskDataset. This is a bit of an ugly hack and not ideal,
        # but should suffice for the time being.
        return DataLoader(
            self.taskset_train,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=identity_collate_fn,
            pin_memory=False,
        )

    def val_dataloader(self):
        if self.taskset_val is not None:
            return DataLoader(
                self.taskset_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=identity_collate_fn,
                pin_memory=False,
            )

    def test_dataloader(self):
        if self.taskset_test is not None:
            return DataLoader(
                self.taskset_test,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=identity_collate_fn,
                pin_memory=False,
            )

    def configure_optimizers(self):
        scheduler_step = 20
        scheduler_decay = 1.0

        optimizer = optim.AdamW(self.parameters(), lr=self.outer_lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step,
            gamma=scheduler_decay,
        )
        return [optimizer], [lr_scheduler]
        # return optimizer

    @staticmethod
    def init_with_fphtr_from_checkpoint(
        fphtr_checkpoint_path: Union[str, Path],
        fphtr_hparams_file: Union[str, Path],
        label_encoder: LabelEncoder,
        *args,
        **kwargs
    ):
        fphtr = LitFullPageHTREncoderDecoder.load_from_checkpoint(
            fphtr_checkpoint_path,
            hparams_file=fphtr_hparams_file,
            label_encoder=label_encoder,
        )
        return MetaHTR(fphtr.model, *args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MetaHTR")
        parser.add_argument("--shots", type=int, default=16)
        parser.add_argument("--ways", type=int, default=8)
        parser.add_argument("--inner_lr", type=float, default=0.0001)
        parser.add_argument("--outer_lr", type=float, default=0.0001)
        return parent_parser


class LitFullPageHTREncoderDecoder(pl.LightningModule):
    model: FullPageHTREncoderDecoder

    """
    Pytorch Lightning module that acting as a wrapper around the
    FullPageHTREncoderDecoder class.

    Using a PL module allows the model to be used in conjunction with a Pytorch
    Lightning Trainer, and takes care of logging relevant metrics to Tensorboard.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        max_seq_len: int = 500,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        encoder_name: str = "resnet18",
        drop_enc: int = 0.5,
        drop_dec: int = 0.5,
        activ_dec: str = "gelu",
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        opt_params = FullPageHTREncoderDecoder.full_page_htr_optimizer_params()
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)
        self.save_hyperparameters(opt_params)
        self.save_hyperparameters(
            "d_model",
            "num_layers",
            "nhead",
            "dim_feedforward",
            "max_seq_len",
            "encoder_name",
            "drop_enc",
            "drop_dec",
            "activ_dec",
        )

        # Initialize the model.
        self.model = FullPageHTREncoderDecoder(
            label_encoder=label_encoder,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            encoder_name=encoder_name,
            drop_enc=drop_enc,
            drop_dec=drop_dec,
            activ_dec=activ_dec,
        )

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None):
        return self.model(imgs, targets)

    def training_step(self, batch, batch_idx, log_results=True):
        imgs, targets = batch
        logits, loss = self.model.forward_teacher_forcing(imgs, targets)
        self.log("train_loss", loss, sync_dist=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        # Forward pass.
        logits, _, loss = self(imgs, targets)
        _, preds = logits.max(-1)

        # Calculate metrics.
        metrics = self.model.calculate_metrics(preds, targets)

        # Log metrics and loss.
        self.log("char_error_rate", metrics["char_error_rate"], prog_bar=True)
        self.log("word_error_rate", metrics["word_error_rate"], prog_bar=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        # `hp_metric` will show up in the Tensorboard hparams tab, used for comparing
        # different models.
        self.log("hp_metric", metrics["char_error_rate"])

        return loss, metrics

    def configure_optimizers(self):
        # By default use the optimizer parameters specified in Singh et al. (2021).
        params = FullPageHTREncoderDecoder.full_page_htr_optimizer_params()
        optimizer_name = params.pop("optimizer_name")
        optimizer = getattr(optim, optimizer_name)(self.parameters(), **params)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitFullPageHTREncoderDecoder")
        parser.add_argument(
            "--encoder",
            type=str,
            choices=["resnet18", "resnet34", "resnet50"],
            default="resnet18",
        )
        parser.add_argument("--d_model", type=int, default=260)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--nhead", type=int, default=4)
        parser.add_argument("--dim_feedforward", type=int, default=1024)
        parser.add_argument(
            "--drop_enc", type=float, default=0.5, help="Encoder dropout."
        )
        parser.add_argument(
            "--drop_dec", type=float, default=0.5, help="Decoder dropout."
        )
        return parent_parser
