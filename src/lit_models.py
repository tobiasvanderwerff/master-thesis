from typing import Callable, Optional, Dict, Union, Any
from pathlib import Path

from models import FullPageHTREncoder, FullPageHTRDecoder, FullPageHTREncoderDecoder
from metrics import CharacterErrorRate, WordErrorRate
from lit_util import TaskDataloader
from util import identity_collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

import learn2learn as l2l


class MetaHTR(pl.LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        taskset_train: l2l.data.TaskDataset,
        ways: int = 8,
        shots: int = 16,
        inner_lr: float = 0.0001,
        outer_lr: float = 0.0001,
        num_inner_steps: int = 1,
        num_workers: int = 0,
        taskset_eval: Optional[l2l.data.TaskDataset] = None,
        taskset_test: Optional[l2l.data.TaskDataset] = None,
    ):
        super().__init__()

        assert num_inner_steps >= 1

        self.taskset_train = taskset_train
        self.taskset_eval = taskset_eval
        self.taskset_test = taskset_test
        self.ways = ways
        self.shots = shots
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.num_workers = num_workers

        # self.automatic_optimization = False  # do manual optimization in training_step()
        self.maml = l2l.algorithms.MAML(model, inner_lr, first_order=False)

        # Pytorch Lightning modules require at minimum one training dataloader, which is
        # why we have to use a small hack to make this work with a `TaskDataset`.
        # Namely, we use a wrapper class `TaskDataLoader`, which does little more than
        # provide a `__getitem__()` and `__len__()` method for a TaskDataset.
        # self.datamodule = TaskDataloader(taskset, ways)

        self.save_hyperparameters(
            "ways", "shots", "inner_lr", "outer_lr", "num_inner_steps"
        )
        self.save_hyperparameters(model.hparams)

    def meta_learn(self, batch, batch_idx, mode="train"):
        outer_loss = 0.0

        imgs, target, writer_ids = batch
        assert imgs.size(0) == 2 * self.ways * self.shots, imgs.size(0)
        # Split the batch into N different writers, where N = ways.
        for task in range(self.ways):
            task_slice = slice(2 * self.shots * task, 2 * self.shots * (task + 1))
            imgs_, target_ = imgs[task_slice], target[task_slice]
            # imgs, target, writer_ids = self.taskset.sample()  # sample a task (writer)

            # Separate data into support/query set.
            support_indices = np.zeros(imgs_.size(0), dtype=bool)
            # Select even indices for support set.
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
                support_loss = learner.module.training_step(
                    [support_imgs, support_tgts], batch_idx, log_results=False
                )
                # Calculate gradients and take an optimization step.
                learner.adapt(support_loss)

            # Outer loop.
            if mode == "train":
                query_loss = learner.module.training_step(
                    [query_imgs, query_tgts], batch_idx, log_results=True
                )
            else:  # val/test
                torch.set_grad_enabled(False)
                query_loss, metrics = learner.module.validation_step(
                    [query_imgs, query_tgts], batch_idx
                )
                for metric, val in metrics.items():
                    self.log(metric, val, prog_bar=True)
                torch.set_grad_enabled(True)
            outer_loss += (1 / self.ways) * query_loss

            # optimizer = self.optimizers(use_pl_optimizer=True)
            # optimizer.zero_grad()
            # Use `manual_backward()` instead of `loss.backward` to automate half
            # precision, etc...
            # self.manual_backward(loss)
            # optimizer.step()
        self.log(f"{mode}_loss", outer_loss, sync_dist=True, prog_bar=False)
        return outer_loss

    def training_step(self, batch, batch_idx):
        loss = self.meta_learn(batch, batch_idx, mode="train")
        # self.log("train_loss", loss, on_epoch=False, on_step=True, logger=True)
        # self.log("train_loss", loss.item(), sync_dist=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation requires finetuning a model in the inner loop, hence we need to
        # enable gradients.
        torch.set_grad_enabled(True)
        loss = self.meta_learn(batch, batch_idx, mode="val")
        torch.set_grad_enabled(False)
        # self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def forward(self, imgs: Tensor):
        """Adapt a model to"""
        logits, _ = self.maml.module(imgs)
        _, preds = logits.max(-1)
        # torch.set_grad_enabled(True)
        # loss = self.meta_learn(imgs, batch_idx=-1, mode="val")
        # torch.set_grad_enabled(False)
        # return self.maml.module(imgs)

    def save_checkpoint(self, filepath: Union[Path, str]):
        torch.save(self.maml.module.state_dict(), filepath)

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
            # num_workers=self.num_workers,
            num_workers=0,  # not sure if higher num_workers is useful in this context
            collate_fn=identity_collate_fn,
            pin_memory=False,
        )
        # return self.taskset
        # return self.datamodule.train_dataloader()
        # return TaskDataloader(self.taskset, self.taskset.num_tasks // self.ways)

    def val_dataloader(self):
        if self.taskset_eval is not None:
            return DataLoader(
                self.taskset_eval,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=identity_collate_fn,
                pin_memory=False,
            )

    def test_dataloader(self):
        if self.taskset_test is not None:
            return DataLoader(
                self.taskset_test,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=identity_collate_fn,
                pin_memory=False,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.maml.parameters(), lr=self.outer_lr)
        return optimizer


class LitFullPageHTREncoderDecoder(pl.LightningModule):
    encoder: FullPageHTREncoder
    decoder: FullPageHTRDecoder
    cer_metric: CharacterErrorRate
    wer_metric: WordErrorRate
    loss_fn: Callable

    def __init__(
        self,
        label_encoder: LabelEncoder,
        encoder_name: str,
        vocab_len: int,
        d_model: int,
        max_seq_len: int,
        eos_tkn_idx: int,
        sos_tkn_idx: int,
        pad_tkn_idx: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        drop_enc: int = 0.1,
        drop_dec: int = 0.5,
        activ_dec: str = "gelu",
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        opt_params = self.full_page_htr_optimizer_params()
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

        model = FullPageHTREncoderDecoder(
            label_encoder=label_encoder,
            encoder_name=encoder_name,
            vocab_len=vocab_len,
            max_seq_len=max_seq_len,
            eos_tkn_idx=eos_tkn_idx,
            sos_tkn_idx=sos_tkn_idx,
            pad_tkn_idx=pad_tkn_idx,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            drop_enc=drop_enc,
            drop_dec=drop_dec,
            activ_dec=activ_dec,
        )

        # Initialize metrics and loss function.
        self.cer_metric = CharacterErrorRate(label_encoder)
        self.wer_metric = WordErrorRate(label_encoder)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.decoder.pad_tkn_idx)

    def forward(self, imgs: Tensor):
        return self(imgs)

    def training_step(self, batch, batch_idx, log_results=True):
        imgs, targets = batch
        memory = self.encoder(imgs)
        logits = self.decoder.decode_teacher_forcing(memory, targets)

        loss = self.loss_fn(logits.transpose(1, 2), targets)
        # if log_results:
        # self.log("train_loss", loss, sync_dist=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits, _ = self(imgs)
        _, preds = logits.max(-1)

        # Calculate metrics and loss.
        cer = self.cer_metric(preds, targets)
        wer = self.wer_metric(preds, targets)
        loss = self.loss_fn(
            logits[:, : targets.size(1), :].transpose(1, 2),
            targets[:, : logits.size(1)],
        )
        metrics = {
            "char_error_rate": cer,
            "word_error_rate": wer,
        }

        # Log metrics and loss.
        # self.log("char_error_rate", cer, prog_bar=True)
        # self.log("word_error_rate", wer)
        # self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        # self.log(
        #     "hp_metric", wer
        # )  # this will show up in the Tensorboard hparams tab, used for comparing different models

        return loss, metrics

    def configure_optimizers(self):
        # By default use the optimizer parameters specified in Singh et al. (2021).
        params = self.full_page_htr_optimizer_params()
        optimizer_name = params.pop("optimizer_name")
        optimizer = getattr(optim, optimizer_name)(self.parameters(), **params)
        return optimizer
