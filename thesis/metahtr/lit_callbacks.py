from typing import Tuple, Optional, Dict

from thesis.lit_callbacks import (
    LogModelPredictionsCallback,
    LogWorstPredictionsCallback,
)
from thesis.metahtr.util import AVAILABLE_MODELS
from thesis.util import TrainMode

from htr.util import LabelEncoder

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback


class LogWorstPredictionsMAML(LogWorstPredictionsCallback):
    def __init__(
        self,
        shots: int,
        ways: int,
        label_encoder: LabelEncoder,
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
                code_size=getattr(pl_module.model, "code_size", None),
                adaptation_num_hidden=getattr(
                    pl_module.model, "adaptation_num_hidden", None
                ),
                inst_mlp_hidden_size=getattr(pl_module, "inst_mlp_hidden_size", None),
                ways=pl_module.model.ways,
                shots=pl_module.model.shots,
                num_inner_steps=pl_module.model.num_inner_steps,
                use_instance_weights=pl_module.use_instance_weights,
            )
        )

        cls = pl_module.__class__
        cls_name = cls.__name__
        assert cls_name in AVAILABLE_MODELS, (
            f"Unrecognized model class: {cls}. " f"Choices: {AVAILABLE_MODELS}"
        )

        model = cls.init_with_base_model_from_checkpoint(**args)
        trainer.model = model


class LogModelPredictionsMAML(LogModelPredictionsCallback):
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
        super().on_train_start(trainer, pl_module)

        # Log the support images once at the start of training.
        if self.val_batch is not None:
            support_imgs, support_targets, _, _, writer = self.val_batch
            self._log_intermediate(
                trainer,
                support_imgs,
                support_targets,
                split=TrainMode.VAL,
                plot_title="support batch",
                plot_suptitle=f"Writer id: {writer}",
            )
        if self.train_batch is not None:
            support_imgs, support_targets, _, _, writer = self.train_batch
            self._log_intermediate(
                trainer,
                support_imgs,
                support_targets,
                split=TrainMode.TRAIN,
                plot_title="support batch",
                plot_suptitle=f"Writer id: {writer}",
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
        else:
            batch = self.val_batch

        # Make predictions.
        support_imgs, support_tgts, query_imgs, query_tgts, writer = batch
        torch.set_grad_enabled(True)
        _, preds, *_ = pl_module(
            *[t.to(device) for t in [support_imgs, support_tgts, query_imgs]]
        )
        torch.set_grad_enabled(False)

        # Log the results.
        self._log_intermediate(
            trainer,
            query_imgs,
            query_tgts,
            preds,
            split=split,
            plot_suptitle=f"Writer id: {writer}",
        )


class LogInstanceSpecificWeights(Callback):
    """Logs the average instance specific weights per ASCII character in a bar plot."""

    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.char_to_avg_inst_weight = None

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.char_to_avg_inst_weight = None

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.char_to_avg_inst_weight = None

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.use_instance_weights:
            char_to_avg_weight = pl_module.char_to_avg_inst_weight
            assert char_to_avg_weight is not None
            self.log_instance_weights(trainer, char_to_avg_weight, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.use_instance_weights:
            char_to_avg_weight = pl_module.char_to_avg_inst_weight
            assert char_to_avg_weight is not None
            self.log_instance_weights(trainer, char_to_avg_weight, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        if pl_module.use_instance_weights:
            char_to_avg_weight = pl_module.char_to_avg_inst_weight
            assert char_to_avg_weight is not None
            self.log_instance_weights(trainer, char_to_avg_weight, "test")

    def log_instance_weights(
        self,
        trainer: "pl.Trainer",
        char_to_avg_weight: Dict[int, float],
        mode: str = "train",
    ):
        # Decode the characters.
        chars, ws = zip(
            *sorted(char_to_avg_weight.items(), key=lambda kv: kv[1], reverse=True)
        )
        chars = self.label_encoder.inverse_transform(chars)

        # Replace special tokens with shorter names to make the plot more readable.
        _chars = []
        _tkn_abbrevs = {"<EOS>": "eos", "<PAD>": "pad", "<SOS>": "sos"}
        for i, c in enumerate(chars):
            if c in _tkn_abbrevs.keys():
                _chars.append(_tkn_abbrevs[c])
            else:
                _chars.append(c)
        chars = _chars

        # Plot the average instance-specific weight per character.
        to_plot = 10
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.bar(chars[:to_plot], ws[:to_plot], align="edge", alpha=0.5)
        plt.grid(True)
        plt.title("Highest")

        plt.subplot(1, 2, 2)
        plt.bar(chars[-to_plot:], ws[-to_plot:], align="edge", alpha=0.5)
        plt.grid(True)
        plt.title("Lowest")

        # Log to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{mode}: average instance-specific weights", fig, trainer.global_step
        )
        plt.close(fig)
