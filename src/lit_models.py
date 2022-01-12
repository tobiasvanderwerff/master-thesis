import time
from typing import Optional, Dict, Union, Tuple, Any
from pathlib import Path

from models import FullPageHTREncoderDecoder
from util import identity_collate_fn, LabelEncoder, LayerWiseLRTransform

import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import learn2learn as l2l
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad


class MetaHTR(pl.LightningModule):
    model: l2l.algorithms.GBML

    _meta_weights = ["model.compute_update", "inst_w_mlp"]

    def __init__(
        self,
        model: FullPageHTREncoderDecoder,
        taskset_train: Union[l2l.data.TaskDataset, Dataset],
        taskset_val: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_test: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        ways: int = 8,
        shots: int = 16,
        outer_lr: float = 0.0001,
        initial_inner_lr: float = 0.001,
        num_inner_steps: int = 1,
        num_workers: int = 0,
        prms_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        assert num_inner_steps >= 1

        self.taskset_train = taskset_train
        self.taskset_val = taskset_val
        self.taskset_test = taskset_test
        self.ways = ways
        self.shots = shots
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.num_workers = num_workers

        self.model = l2l.algorithms.GBML(
            model,
            transform=LayerWiseLRTransform(initial_inner_lr),
            lr=1.0,  # this lr is replaced by a learnable one
            first_order=False,
            allow_nograd=True,
        )
        self.inst_w_mlp = nn.Sequential(  # instance-specific weight MLP
            # TODO: how big to make the dense layers?
            nn.Linear(
                model.decoder.clf.in_features * model.decoder.clf.out_features * 2, 256
            ),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.save_hyperparameters(
            "ways",
            "shots",
            "outer_lr",
            "initial_inner_lr",
            "num_inner_steps",
        )
        if prms_to_log is not None:
            self.save_hyperparameters(prms_to_log)

    def meta_learn(self, batch, mode="train") -> Tensor:
        outer_loss = 0.0

        imgs, target, writer_ids = batch
        writer_ids_uniq = writer_ids.unique().tolist()

        assert mode in ["train", "val", "test"]
        assert imgs.size(0) >= 2 * self.ways * self.shots, imgs.size(0)
        assert len(writer_ids_uniq) == self.ways

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

            # Calling `model.clone()` allows updating the module while still allowing
            # computation of derivatives of the new modules' parameters w.r.t. the
            # original parameters.
            learner = self.model.clone()

            learner.eval()
            # set_norm_layers_to_train(learner)
            # learner.train()

            # Inner loop.
            # autograd_hack.add_hooks(learner)
            for _ in range(self.num_inner_steps):
                _, support_loss_unreduced = learner.module.forward_teacher_forcing(
                    support_imgs, support_tgts
                )
                # mean_loss.backward(inputs=learner.module.decoder.clf.weight)
                # autograd_hack.compute_grad1(learner, loss_type="mean")

                instance_weights = self.calculate_instance_specific_weights(
                    learner, support_loss_unreduced
                )
                support_loss = torch.mean(
                    support_loss_unreduced.view(-1) * instance_weights
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
                query_loss = torch.mean(query_loss)
            else:  # val/test
                with torch.inference_mode():
                    _, preds, query_loss = learner(query_imgs, query_tgts)
                    query_loss = torch.mean(query_loss)

                # Log metrics.
                metrics = learner.module.calculate_metrics(preds, query_tgts)
                for metric, val in metrics.items():
                    self.log(metric, val, prog_bar=True)
            outer_loss += (1 / self.ways) * query_loss

        return outer_loss

    def calculate_instance_specific_weights(
        self, learner: l2l.algorithms.GBML, loss_unreduced: Tensor
    ) -> Tensor:
        """
        Calculates instance-specific weights, based on the per-instance gradient
        w.r.t to the final classifcation layer.

        Args:
            learner (l2l.algorithms.GBML): learn2learn GBML learner
            loss_unreduced (Tensor): tensor of shape (B*T,), where B = batch size and
                T = maximum sequence length in the batch, containing the per-instance
                loss, i.e. the loss for each decoding time step.

        Returns:
            Tensor of shape (B*T,), containing the instance specific weights
        """
        grad_inputs = []
        start_time = time.time()

        mean_loss_grad = grad(
            torch.mean(loss_unreduced),
            learner.module.decoder.clf.weight,
            create_graph=True,
            retain_graph=True,
        )[0]
        # It is not ideal to have to compute gradients like this in a loop - which
        # loses the benefit of parallelization -, but unfortunately Pytorch does not
        # provide any native functonality for calculating per-example gradients.
        for instance_loss in loss_unreduced.view(-1):
            instance_grad = grad(
                instance_loss,
                learner.module.decoder.clf.weight,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_inputs.append(
                torch.cat([instance_grad.flatten(), mean_loss_grad.flatten()])
            )
        print(
            f"Time (s) spent calculating per-example gradients: "
            f"{time.time() - start_time:.2f}"
        )
        grad_inputs = torch.stack(grad_inputs, 0).detach()
        instance_weights = self.inst_w_mlp(grad_inputs)
        return instance_weights

    @property
    def encoder(self):
        return self.model.module.encoder

    @property
    def decoder(self):
        return self.model.module.decoder

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
        learner = self.model.clone()
        learner.eval()
        # set_norm_layers_to_train(learner)
        # learner.train()

        # For some reason using an autograd context manager like torch.enable_grad()
        # here does not work, perhaps due to some unexpected interaction between
        # Pytorch Lightning and the learn2learn lib. Therefore gradient
        # calculation should be set beforehand, outside of the current function.
        _, adaptation_loss_unreduced = learner.module.forward_teacher_forcing(
            adaptation_imgs, adaptation_targets
        )
        instance_weights = self.calculate_instance_specific_weights(
            learner, adaptation_loss_unreduced
        )
        adaptation_loss = torch.mean(
            adaptation_loss_unreduced.view(-1) * instance_weights
        )
        # Adapt the model.
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

    def freeze_all_layers_except_classifier(self):
        for n, p in self.named_parameters():
            if not n.split(".")[-2] == "clf":
                p.requires_grad = False

    @staticmethod
    def init_with_fphtr_from_checkpoint(
        checkpoint_path: Union[str, Path],
        fphtr_hparams_file: Union[str, Path],
        label_encoder: LabelEncoder,
        load_meta_weights: bool = False,
        fphtr_params_to_log: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        # Load FPHTR model.
        fphtr = LitFullPageHTREncoderDecoder.load_from_checkpoint(
            checkpoint_path,
            hparams_file=fphtr_hparams_file,
            # `strict=False` because MAML checkpoints contain additional parameters
            # in the state_dict (namely the learnable inner loop learning rates), which
            # should be ignored when loading FPHTR.
            strict=False,
            label_encoder=label_encoder,
            params_to_log=fphtr_params_to_log,
            loss_reduction="none",  # necessary for instance-specific loss weights
        )

        model = MetaHTR(fphtr.model, *args, **kwargs)

        if load_meta_weights:
            # Load weights specific to the meta-learning algorithm.
            ckpt = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            for n, p in ckpt["state_dict"].items():
                if any(n.startswith(wn) for wn in MetaHTR._meta_weights):
                    with torch.no_grad():
                        model.state_dict()[n] = p
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MetaHTR")
        parser.add_argument("--shots", type=int, default=16)
        parser.add_argument("--ways", type=int, default=8)
        parser.add_argument("--outer_lr", type=float, default=0.0001)
        parser.add_argument("--initial_inner_lr", type=float, default=0.001)
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
        learning_rate: float = 0.0002,
        max_seq_len: int = 500,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        encoder_name: str = "resnet18",
        drop_enc: int = 0.5,
        drop_dec: int = 0.5,
        activ_dec: str = "gelu",
        loss_reduction: str = "mean",
        vocab_len: Optional[int] = None,  # if not specified len(label_encoder) is used
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        self.learning_rate = learning_rate
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)
        self.save_hyperparameters(
            "learning_rate",
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
            vocab_len=vocab_len,
            loss_reduction=loss_reduction,
        )

        self.all_logits = None
        self.all_targets = None

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
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("LitFullPageHTREncoderDecoder")
        parser.add_argument("--learning_rate", type=float, default=0.0002)
        parser.add_argument("--encoder", type=str, default="resnet18",
                            choices=["resnet18", "resnet34", "resnet50"])
        parser.add_argument("--d_model", type=int, default=260)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--nhead", type=int, default=4)
        parser.add_argument("--dim_feedforward", type=int, default=1024)
        parser.add_argument("--drop_enc", type=float, default=0.5,
                            help="Encoder dropout.")
        parser.add_argument("--drop_dec", type=float, default=0.5,
                            help="Decoder dropout.")
        return parent_parser
        # fmt: on
