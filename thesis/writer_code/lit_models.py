from typing import Optional, Dict, Union, Tuple, Any
from pathlib import Path

from thesis.writer_code.models import WriterCodeAdaptiveModel
from thesis.util import split_batch_for_adaptation, identity_collate_fn

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead
from htr.models.lit_models import LitShowAttendRead, LitFullPageHTREncoderDecoder
from htr.util import LabelEncoder

import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import learn2learn as l2l
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from thesis.writer_code.util import WriterEmbeddingType


class LitWriterCodeAdaptiveModel(pl.LightningModule):

    meta_weights = ["writer_embs", "adaptation"]

    def __init__(
        self,
        model: nn.Module,
        feature_size: int,
        num_writers: int,
        taskset_train: Union[l2l.data.TaskDataset, Dataset],
        taskset_val: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_test: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        val_batch_size: int = 64,
        writer_emb_size: int = 64,
        writer_emb_type: WriterEmbeddingType = WriterEmbeddingType.LEARNED,
        adapt_num_hidden: int = 1000,
        ways: int = 8,
        shots: int = 8,
        learning_rate: float = 0.0001,
        learning_rate_emb: float = 0.0001,
        weight_decay: float = 0.0001,
        adaptation_opt_steps: int = 1,
        grad_clip: Optional[float] = None,
        use_adam_for_adaptation: bool = False,
        use_cosine_lr_scheduler: bool = False,
        num_workers: int = 0,
        num_epochs: Optional[int] = None,
        prms_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        """
        Args:
            TODO
            model (nn.Module): base model
            feature_size (int): size of the feature vectors to adapt, e.g. the output
                feature vectors of a CNN
            num_writers (int): number of writers in the training set
            writer_emb_type (WriterEmbeddingType): type of writer embedding used
        """
        super().__init__()

        assert not (use_cosine_lr_scheduler and num_epochs is None), (
            "When using cosine learning rate scheduler, specify `num_epochs` to "
            "configure the learning rate decay properly."
        )
        assert isinstance(model, (FullPageHTREncoderDecoder, ShowAttendRead))

        self.model = model
        self.feature_size = feature_size
        self.num_writers = num_writers
        self.taskset_train = taskset_train
        self.taskset_val = taskset_val
        self.taskset_test = taskset_test
        self.val_batch_size = val_batch_size
        self.writer_emb_size = writer_emb_size
        self.writer_emb_type = writer_emb_type
        self.adapt_num_hidden = adapt_num_hidden
        self.ways = ways
        self.shots = shots
        self.learning_rate = learning_rate
        self.learning_rate_emb = learning_rate_emb
        self.weight_decay = weight_decay
        self.adaptation_opt_steps = adaptation_opt_steps
        self.grad_clip = grad_clip
        self.use_adam_for_adaptation = use_adam_for_adaptation
        self.use_cosine_lr_schedule = use_cosine_lr_scheduler
        self.num_workers = num_workers
        self.num_epochs = num_epochs

        self.ignore_index = model.pad_tkn_idx
        self.automatic_optimization = False

        self.adaptive_model = WriterCodeAdaptiveModel(
            base_model=model,
            d_model=feature_size,
            emb_size=writer_emb_size,
            num_hidden=adapt_num_hidden,
            num_writers=num_writers,
            learning_rate_emb=learning_rate_emb,
            embedding_type=writer_emb_type,
            adaptation_opt_steps=adaptation_opt_steps,
            use_adam_for_adaptation=use_adam_for_adaptation,
        )

        self.save_hyperparameters(
            "writer_emb_type",
            "writer_emb_size",
            "adapt_num_hidden",
            "ways",
            "shots",
            "learning_rate",
        )
        if prms_to_log is not None:
            self.save_hyperparameters(prms_to_log)

    def training_step(self, batch, batch_idx):
        imgs, target, writer_ids = batch
        _, _, loss = self.adaptive_model(imgs, target, writer_ids, mode="train")
        self.opt_step(loss)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def opt_step(self, loss: Tensor, inputs: Optional[Tensor] = None):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss.backward(inputs=inputs)
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
        writer_batches = split_batch_for_adaptation(
            batch, self.ways, self.shots, limit_num_samples_per_task=self.val_batch_size
        )
        for adapt_imgs, adapt_tgts, query_imgs, query_tgts in writer_batches:
            # TODO: see if this can be processed in a single batch (multiple writers)
            torch.set_grad_enabled(True)
            _, preds, query_loss = self(
                adapt_imgs, adapt_tgts, query_imgs, query_tgts, mode="val"
            )
            torch.set_grad_enabled(False)

            # Log metrics.
            self.model.cer_metric(preds, query_tgts)
            self.model.wer_metric(preds, query_tgts)
            self.log("char_error_rate", self.model.cer_metric, prog_bar=False)
            self.log("word_error_rate", self.model.wer_metric, prog_bar=True)

            loss += query_loss * query_imgs.size(0)
            n_samples += query_imgs.size(0)
        loss /= n_samples
        self.log(f"{mode}_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def forward(
        self,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
        inference_imgs: Tensor,
        inference_tgts: Optional[Tensor] = None,
        mode: str = "train",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self.eval()
        return self.adaptive_model(
            adaptation_imgs,
            adaptation_targets,
            inference_imgs,
            inference_tgts,
            mode=mode,
        )

    def training_epoch_end(self, epoch_outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

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
            feature_size = base_model.encoder.resnet_out_features
        else:  # SAR
            base_model = LitShowAttendRead.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
            )
            feature_size = base_model.rnn_encoder.input_size

        model = LitWriterCodeAdaptiveModel(
            base_model.model, feature_size, *args, **kwargs
        )

        if load_meta_weights:
            # Load weights specific to the meta-learning algorithm.
            loaded = []
            ckpt = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            for n, p in ckpt["state_dict"].items():
                if any(
                    n.startswith(wn) for wn in LitWriterCodeAdaptiveModel.meta_weights
                ):
                    with torch.no_grad():
                        model.state_dict()[n][:] = p
                    loaded.append(n)
            print(f"Loaded meta weights: {loaded}")
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitWriterCodeAdaptiveModel")
        parser.add_argument(
            "--writer_emb_size",
            type=int,
            default=64,
            help="Size of the writer embeddings for adaptation.",
        )
        parser.add_argument(
            "--writer_emb_type",
            type=str,
            default="learned",
            choices=["learned", "transformed"],
            help="Type of writer embedding to use.",
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
            default=0.0001,
            help="Learning rate used for creating writer embeddings during val/test",
        )
        parser.add_argument(
            "--adaptation_opt_steps",
            type=int,
            default=1,
            help="Number of optimization steps to perform for "
            "training a new writer code during val/test.",
        )
        parser.add_argument(
            "--use_adam_for_adaptation",
            action="store_true",
            default=False,
            help="Use Adam during val/test for training new writer codes.",
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
