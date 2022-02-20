from typing import Optional, Dict, Union, Tuple, Any, List, Sequence
from pathlib import Path
from collections import defaultdict

from thesis.metahtr.models import MetaHTR, MAMLHTR
from thesis.util import identity_collate_fn, TrainMode

from htr.models.sar.sar import ShowAttendRead
from htr.models.lit_models import LitShowAttendRead, LitFullPageHTREncoderDecoder
from htr.util import LabelEncoder

import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import learn2learn as l2l
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class LitMAMLHTR(pl.LightningModule):
    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        taskset_train: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_val: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_test: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        outer_lr: float = 0.0001,
        val_batch_size: int = 64,
        grad_clip: Optional[float] = None,
        num_workers: int = 0,
        num_epochs: Optional[int] = None,
        use_cosine_lr_scheduler: bool = False,
        prms_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
        **kwargs,
    ):
        """
        Args:
            TODO
            ...
            use_cosine_lr_scheduler (bool): whether to use a cosine annealing
                scheduler to decay the learning rate from its initial value.
            num_epochs (Optional[int]): number of epochs the model will be trained.
                This is only used if `use_cosine_lr_scheduler` is set to True.
        """
        super().__init__()

        assert not (use_cosine_lr_scheduler and num_epochs is None), (
            "When using cosine learning rate scheduler, specify `num_epochs` to "
            "configure the learning rate decay properly."
        )

        self.taskset_train = taskset_train
        self.taskset_val = taskset_val
        self.taskset_test = taskset_test
        self.outer_lr = outer_lr
        self.val_batch_size = val_batch_size
        self.grad_clip = grad_clip
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.use_cosine_lr_schedule = use_cosine_lr_scheduler

        self.model = None
        if base_model is not None:
            self.model = MAMLHTR(
                base_model=base_model,
                **kwargs,
            )

        self.automatic_optimization = False
        self.use_instance_weights = False
        self.char_to_avg_inst_weight = None

        self.save_hyperparameters(
            "ways",
            "shots",
            "outer_lr",
            "num_inner_steps",
            "use_instance_weights",
        )
        if prms_to_log is not None:
            self.save_hyperparameters(prms_to_log)

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        opt = self.optimizers()
        opt.zero_grad()
        outer_loss, inner_loss, inst_ws = self.model.meta_learn(
            batch, mode=TrainMode.TRAIN
        )
        if self.grad_clip is not None:
            self.clip_gradients(opt, self.grad_clip, "norm")
        opt.step()
        self.log("train_loss_outer", outer_loss, sync_dist=True, prog_bar=False)
        self.log(f"train_loss_inner", inner_loss, sync_dist=True, prog_bar=False)
        return {"loss": outer_loss, "char_to_inst_weights": inst_ws}

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.VAL)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.TEST)

    def val_or_test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.VAL
    ):
        # val/test requires finetuning a model in the inner loop, hence we need to
        # enable gradients.
        torch.set_grad_enabled(True)
        outer_loss, inner_loss, inst_ws = self.model.meta_learn(batch, mode=mode)
        torch.set_grad_enabled(False)
        self.log(
            f"{mode.name.lower()}_loss_outer", outer_loss, sync_dist=True, prog_bar=True
        )
        self.log(
            f"{mode.name.lower()}_loss_inner",
            inner_loss,
            sync_dist=True,
            prog_bar=False,
        )
        self.log("char_error_rate", self.model.gbml.module.cer_metric, prog_bar=True)
        self.log("word_error_rate", self.model.gbml.module.wer_metric, prog_bar=True)
        return {"loss": outer_loss, "char_to_inst_weights": inst_ws}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_epoch_end(self, epoch_outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def validation_epoch_end(self, epoch_outputs):
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def test_epoch_end(self, epoch_outputs):
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def aggregate_epoch_instance_weights(
        self, training_epoch_outputs: Sequence[Dict[Any, Any]]
    ):
        char_to_weights, char_to_avg_weight = defaultdict(list), defaultdict(float)
        # Aggregate all instance-specific weights.
        for dct in training_epoch_outputs:
            char_to_ws = dct["char_to_inst_weights"]
            for c_idx, ws in char_to_ws.items():
                char_to_weights[c_idx].extend(ws)
        # Calculate average instance weight per class for logging purposes.
        for c_idx, ws in char_to_weights.items():
            char_to_avg_weight[c_idx] = np.mean(ws)
        self.char_to_avg_inst_weight = char_to_avg_weight

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
        optimizer = optim.Adam(self.parameters(), lr=self.outer_lr)
        if self.use_cosine_lr_schedule:
            num_epochs = self.num_epochs or 20
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-06,  # final learning rate
                verbose=True,
            )
            return [optimizer], [lr_scheduler]
        return optimizer

    @staticmethod
    def init_with_base_model_from_checkpoint(
        model_arch: str,
        checkpoint_path: Union[str, Path],
        model_hparams_file: Union[str, Path],
        label_encoder: LabelEncoder,
        load_meta_weights: bool = False,
        model_params_to_log: Optional[Dict[str, Any]] = None,
        metahtr: bool = False,
        **kwargs,
    ):
        assert model_arch in ["fphtr", "sar"], "Invalid base model architecture."

        if model_arch == "fphtr":
            # Load FPHTR model.
            base_model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
                loss_reduction="none",  # necessary for instance-specific loss weights
            )
            num_clf_weights = (
                base_model.decoder.clf.in_features * base_model.decoder.clf.out_features
            )
        else:  # SAR
            base_model = LitShowAttendRead.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
                loss_reduction="none",  # necessary for instance-specific loss weights
            )
            num_clf_weights = (
                base_model.lstm_decoder.prediction.in_features
                * base_model.lstm_decoder.prediction.out_features
            )

        if metahtr:
            model = LitMetaHTR(
                base_model=base_model.model, num_clf_weights=num_clf_weights, **kwargs
            )
        else:
            model = LitMAMLHTR(base_model=base_model.model, **kwargs)

        if load_meta_weights:
            # Load weights specific to the meta-learning algorithm.
            loaded = []
            ckpt = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            for n, p in ckpt["state_dict"].items():
                if any(n.startswith("model." + wn) for wn in MetaHTR.meta_weights):
                    with torch.no_grad():
                        model.state_dict()[n][:] = p
                    loaded.append(n)
            print(f"Loaded meta weights: {loaded}")
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MAML")
        parser.add_argument("--shots", type=int, default=16)
        parser.add_argument("--ways", type=int, default=8)
        parser.add_argument("--outer_lr", type=float, default=0.0001)
        parser.add_argument("--grad_clip", type=float, help="Max. gradient norm.")
        parser.add_argument("--num_inner_steps", type=int, default=1)
        parser.add_argument(
            "--use_cosine_lr_scheduler",
            action="store_true",
            default=False,
            help="Use a cosine annealing scheduler to " "decay the learning rate.",
        )
        parser.add_argument(
            "--use_batch_stats_for_batchnorm",
            action="store_true",
            default=False,
            help="Use batch statistics over stored statistics for batchnorm layers.",
        )
        parser.add_argument(
            "--use_dropout",
            action="store_true",
            default=False,
            help="Use dropout in the outer loop",
        )
        parser.add_argument(
            "--no_instance_weights",
            action="store_true",
            default=False,
            help="Do not use instance-specific weights proposed in the "
            "MetaHTR paper.",
        )
        parser.add_argument(
            "--freeze_batchnorm_gamma",
            action="store_true",
            default=False,
            help="Freeze gamma (scaling factor) for all batchnorm layers.",
        )
        return parent_parser


class LitMetaHTR(LitMAMLHTR):
    def __init__(
        self,
        base_model: nn.Module,
        num_clf_weights: int,
        inst_mlp_hidden_size: int = 8,
        initial_inner_lr: float = 0.001,
        use_instance_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_clf_weights = num_clf_weights
        self.inst_mlp_hidden_size = inst_mlp_hidden_size
        self.initial_inner_lr = initial_inner_lr
        self.use_instance_weights = use_instance_weights

        self.model = MetaHTR(
            base_model=base_model,
            num_clf_weights=num_clf_weights,
            inst_mlp_hidden_size=inst_mlp_hidden_size,
            initial_inner_lr=initial_inner_lr,
            use_instance_weights=use_instance_weights,
            **kwargs,
        )

        self.save_hyperparameters("inst_mlp_hidden_size", "initial_inner_lr")

    @staticmethod
    def init_with_base_model_from_checkpoint(*args, **kwargs):
        return LitMAMLHTR.init_with_base_model_from_checkpoint(
            metahtr=True, *args, **kwargs
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MetaHTR")
        parser.add_argument("--inst_mlp_hidden_size", type=int, default=8)
        parser.add_argument("--initial_inner_lr", type=float, default=0.001)
        return LitMAMLHTR.add_model_specific_args(parent_parser)
