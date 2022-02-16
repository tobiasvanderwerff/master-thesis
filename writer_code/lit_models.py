from typing import Optional, Dict, Union, Tuple, Any, List
from pathlib import Path

from metahtr.util import identity_collate_fn

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead
from htr.models.lit_models import LitShowAttendRead, LitFullPageHTREncoderDecoder
from htr.util import LabelEncoder

from util import BatchNorm1dPermute

import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import learn2learn as l2l
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class WriterCodeAdaptiveModel(pl.LightningModule):

    meta_weights = ["writer_embs", "adaptation_layers"]

    def __init__(
        self,
        model: nn.Module,
        feature_size: int,
        num_writers: int,
        taskset_train: Union[l2l.data.TaskDataset, Dataset],
        taskset_val: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_test: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        writer_emb_method: str = "concat",
        writer_emb_size: int = 64,
        adapt_num_hidden: int = 1000,
        ways: int = 8,
        shots: int = 8,
        learning_rate: float = 0.0001,
        learning_rate_emb: float = 0.001,
        weight_decay: float = 0.0001,
        grad_clip: Optional[float] = None,
        use_cosine_lr_scheduler: bool = False,
        num_workers: int = 0,
        num_epochs: Optional[int] = None,
        prms_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        """
        Args:
            model (nn.Module): base model
            feature_size (int): size of the feature vectors to adapt, e.g. the output
                feature vectors of a CNN
            num_writers (int): number of writers in the training set
        """
        super().__init__()

        assert not (use_cosine_lr_scheduler and num_epochs is None), (
            "When using cosine learning rate scheduler, specify `num_epochs` to "
            "configure the learning rate decay properly."
        )
        assert writer_emb_method in ["sum", "concat", "transform"]
        assert isinstance(model, (FullPageHTREncoderDecoder, ShowAttendRead))

        self.model = model
        self.feature_size = feature_size
        self.num_writers = num_writers
        self.taskset_train = taskset_train
        self.taskset_val = taskset_val
        self.taskset_test = taskset_test
        self.writer_emb_method = writer_emb_method
        self.writer_emb_size = writer_emb_size
        self.adapt_num_hidden = adapt_num_hidden
        self.ways = ways
        self.shots = shots
        self.learning_rate = learning_rate
        self.learning_rate_emb = learning_rate_emb
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_cosine_lr_schedule = use_cosine_lr_scheduler
        self.num_workers = num_workers
        self.num_epochs = num_epochs

        self.ignore_index = model.pad_tkn_idx
        self.automatic_optimization = False

        self.freeze()  # freeze the original model

        in_size = feature_size
        hidden_size = adapt_num_hidden
        if writer_emb_method == "sum":
            assert feature_size == writer_emb_size
            raise NotImplementedError("Sum not implemented yet.")
        elif writer_emb_method == "concat":
            in_size = in_size + writer_emb_size
            hidden_size = hidden_size + writer_emb_size
        elif writer_emb_method == "transform":
            raise NotImplementedError("Transform not implemented yet.")

        self.writer_embs = nn.Embedding(num_writers, writer_emb_size)
        self.adaptation_layers = nn.ModuleList(
            [
                # TODO: add dropout/batchnorm?
                nn.Sequential(
                    nn.Linear(in_size, adapt_num_hidden),
                    nn.ReLU(inplace=True),
                    BatchNorm1dPermute(adapt_num_hidden),
                ),
                nn.Sequential(
                    nn.Linear(hidden_size, adapt_num_hidden),
                    nn.ReLU(inplace=True),
                    BatchNorm1dPermute(adapt_num_hidden),
                ),
                nn.Sequential(nn.Linear(hidden_size, feature_size)),
            ]
        )

        assert all(p.requires_grad for p in self.writer_embs.parameters())
        assert all(p.requires_grad for p in self.adaptation_layers.parameters())
        assert model.loss_fn.reduction == "mean"

        self.save_hyperparameters(
            "writer_emb_method",
            "writer_emb_size",
            "adapt_num_hidden",
            "ways",
            "shots",
            "learning_rate",
        )
        if prms_to_log is not None:
            self.save_hyperparameters(prms_to_log)

    def base_model_forward(
        self,
        imgs: Tensor,
        writer_emb: Tensor,
        target: Optional[Tensor] = None,
        teacher_forcing: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass using writer codes."""
        if isinstance(self.model, FullPageHTREncoderDecoder):
            features = self.model.encoder(imgs)
            features = self.adapt_features(features, writer_emb)
            if teacher_forcing:
                logits = self.model.decoder.decode_teacher_forcing(features, target)
            else:
                logits, _ = self.model.decoder(features)
        else:  # SAR
            features = self.model.resnet_encoder(imgs)
            features = self.adapt_features(features, writer_emb)
            h_holistic = self.model.lstm_encoder(features)
            if teacher_forcing:
                logits = self.model.lstm_decoder.forward_teacher_forcing(
                    features, h_holistic, target
                )
            else:
                logits, _ = self.model.lstm_decoder(features, h_holistic)
        loss = None
        if target is not None:
            loss = self.model.loss_fn(
                logits[:, : target.size(1), :].transpose(1, 2),
                target[:, : logits.size(1)],
            )
        return logits, loss

    def adapt_features(self, features: Tensor, writer_emb: Tensor) -> Tensor:
        """
        Adapt features based on a writer code (embedding).

        Features are adapted by passing them through an adaptation network,
        along with a writer embedding. The writer embedding is added as
        additional input for each layer of the adaptation network.

        Args:
             features (Tensor of shape (N, M, d_model)): features to adapt
             writer_emb (Tensor of shape (N, emb_size)): writer embedding for each
                sample, used for adapting the features
        Returns:
            Tensor of shape (N, M, d_model), containing transformed features
        """
        writer_emb = writer_emb.unsqueeze(1)
        writer_emb = writer_emb.expand(features.size(0), features.size(1), -1)
        res = features
        for layer in self.adaptation_layers:
            if self.writer_emb_method == "concat":
                res = torch.cat((res, writer_emb), -1)
            res = layer(res)
        return res + features

    def training_step(self, batch, batch_idx):
        imgs, target, writer_ids = batch
        wrtr_emb = self.writer_embs(writer_ids)  # (N, emb_size)
        logits, loss = self.base_model_forward(
            imgs, wrtr_emb, target, teacher_forcing=True
        )
        self.opt_step(loss)
        self.log("train_loss", loss, sync_dist=True, prog_bar=False)
        return loss

    def opt_step(self, loss: Tensor, inputs: Optional[Tensor] = None):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss, inputs=inputs)
        if self.grad_clip is not None:
            self.clip_gradients(optimizer, self.grad_clip, "norm")
        optimizer.step()

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode="test")

    def val_or_test_step(self, batch, mode="val"):
        """
        Val/test step. The difference with a train step:

        - A new writer code is created based on a small batch of data, named the
          "adaptation batch". This is necessary because the writers in the val/test
          set do not overlap with those in the train set, thus necessitating new
          writer codes.
        - Teacher forcing is not used.
        """
        loss, n_samples = 0, 0
        writer_batches = self.split_batch_for_adaptation(batch)
        for adapt_imgs, adapt_tgts, query_imgs, query_tgts in writer_batches:
            # TODO: see if this can be processed in a single batch (multiple writers)
            torch.set_grad_enabled(True)
            _, preds, query_loss = self(adapt_imgs, adapt_tgts, query_imgs, query_tgts)
            torch.set_grad_enabled(False)

            # Log metrics.
            self.model.cer_metric(preds, query_tgts)
            self.model.wer_metric(preds, query_tgts)
            self.log("char_error_rate", self.model.cer_metric, prog_bar=True)
            self.log("word_error_rate", self.model.wer_metric, prog_bar=True)

            loss += query_loss * query_imgs.size(0)
            n_samples += query_imgs.size(0)
        loss /= n_samples
        self.log(f"{mode}_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def split_batch_for_adaptation(
        self, batch
    ) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        imgs, target, writer_ids = batch
        writer_ids_uniq = writer_ids.unique().tolist()

        assert imgs.size(0) >= 2 * self.ways * self.shots, imgs.size(0)
        assert (
            len(writer_ids_uniq) == self.ways
        ), f"{len(writer_ids_uniq)} vs {self.ways}"

        # Split the batch into N different writers, where N = ways.
        writer_batches = []
        for task in range(self.ways):  # tasks correspond to different writers
            wrtr_id = writer_ids_uniq[task]
            task_slice = writer_ids == wrtr_id
            imgs_, target_, writer_ids_ = (
                imgs[task_slice],
                target[task_slice],
                writer_ids[task_slice],
            )

            # Separate data into support/query set.
            adaptation_indices = np.zeros(imgs_.size(0), dtype=bool)
            # Select first k even indices for adaptation set.
            adaptation_indices[np.arange(self.shots) * 2] = True
            # Select remaining indices for query set.
            query_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_imgs, adaptation_tgts = (
                imgs_[adaptation_indices],
                target_[adaptation_indices],
            )
            query_imgs, query_tgts = imgs_[query_indices], target_[query_indices]
            writer_batches.append(
                (adaptation_imgs, adaptation_tgts, query_imgs, query_tgts)
            )

        return writer_batches

    def training_epoch_end(self, epoch_outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def forward(
        self,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
        inference_imgs: Tensor,
        inference_tgts: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Adapt on a set of images for a particular writer and run inference on another set.

        Args:
            adaptation_imgs (Tensor): images to do adaptation on
            adaptation_targets (Tensor): targets for `adaptation_imgs`
            inference_imgs (Tensor): images to make predictions on
            inference_tgts (Optional[Tensor]): targets for `inference_imgs`
        Returns:
            predictions on `inference_imgs`, in the form of a 3-tuple:
                - logits, obtained at each time step during decoding
                - sampled class indices, i.e. model predictions, obtained by applying
                      greedy decoding (argmax on logits) at each time step
                - mean loss
        """
        self.eval()

        # Train a writer-specific code.
        writer_code = self.new_writer_code(adaptation_imgs, adaptation_targets)

        # Run inference using the writer code.
        with torch.inference_mode():
            logits, loss = self.base_model_forward(
                inference_imgs, writer_code, inference_tgts, teacher_forcing=False
            )
        sampled_ids = logits.argmax(-1)

        return logits, sampled_ids, loss

    def new_writer_code(
        self, adaptation_imgs: Tensor, adaptation_targets: Tensor
    ) -> Tensor:
        """
        Create a new writer code (embedding) based on a batch of examples for a writer.

        The writer code is created by running a single forward/backward
        pass on the batch of data in order to initialize a new writer embedding.
        """
        writer_emb = torch.empty(1, self.writer_emb_size).to(adaptation_imgs.device)
        writer_emb.normal_()  # mean 0, std 1
        writer_emb.requires_grad = True
        # writer_emb = torch.nn.Parameter(writer_emb, requires_grad=True)

        # for p in self.adaptation_layers.parameters():
        #     p.requires_grad = False
        # assert sum(int(p.requires_grad) for p in self.parameters()) == 1, \
        #     "Only the new writer embedding should receive gradients."

        # TODO: is only one gradient update enough?
        _, loss = self.base_model_forward(
            adaptation_imgs, writer_emb, adaptation_targets
        )
        old_weight = writer_emb.data

        # Take a gradient step.
        self.manual_backward(loss, inputs=writer_emb)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(writer_emb, self.grad_clip)
        lr = self.learning_rate_emb
        writer_emb = (writer_emb - lr * writer_emb.grad.data).detach()

        assert (writer_emb.data != old_weight).any()

        return writer_emb

    def set_batchnorm_layers_train(self, training: bool = True):
        _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d)
        for m in self.modules():
            if isinstance(m, _batchnorm_layers):
                m.training = training

    def set_dropout_layers_train(self, training: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.training = training

    def batchnorm_reset_running_stats(self):
        _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d)
        for m in self.modules():
            if isinstance(m, _batchnorm_layers):
                m.reset_running_stats()

    def freeze_all_layers_except_classifier(self):
        for n, p in self.named_parameters():
            p.requires_grad = False
        self.model.decoder.clf.requires_grad_(True)

    def freeze_batchnorm_weights(self, freeze_bias=False):
        """
        For all normalization layers (of the form x * w + b), freeze w,
        and optionally the bias.
        """
        _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
        for m in self.modules():
            if isinstance(m, _batchnorm_layers):
                m.weight.requires_grad = False
                if freeze_bias:
                    m.bias.requires_grad = False

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
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
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
        *args,
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
            )
            feature_size = base_model.encoder.d_model
        else:  # SAR
            base_model = LitShowAttendRead.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
            )
            feature_size = base_model.rnn_encoder.input_size

        model = WriterCodeAdaptiveModel(base_model.model, feature_size, *args, **kwargs)

        if load_meta_weights:
            pass
            # Load weights specific to the meta-learning algorithm.
            loaded = []
            ckpt = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            for n, p in ckpt["state_dict"].items():
                if any(n.startswith(wn) for wn in WriterCodeAdaptiveModel.meta_weights):
                    with torch.no_grad():
                        model.state_dict()[n][:] = p
                    loaded.append(n)
            print(f"Loaded meta weights: {loaded}")
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WriterCodeAdaptiveModel")
        parser.add_argument(
            "--writer_emb_size",
            type=int,
            default=64,
            help="Size of the writer embeddings for adaptation.",
        )
        parser.add_argument(
            "--writer_emb_method",
            type=str,
            default="concat",
            choices=["add", "concat", "transform"],
            help="How to inject writer embeddings into the model.",
        )
        parser.add_argument(
            "--adapt_num_hidden",
            type=int,
            default=1000,
            help="Number of features for the hidden layers of the " "adaptation MLP",
        )
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument(
            "--learning_rate_emb",
            type=float,
            default=0.001,
            help="Learning rate used for creating writer embeddings " "during val/test",
        )
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--shots", type=int, default=8)
        parser.add_argument("--ways", type=int, default=8)
        parser.add_argument(
            "--use_cosine_lr_scheduler",
            action="store_true",
            default=False,
            help="Use a cosine annealing scheduler to " "decay the learning rate.",
        )
        return parent_parser
