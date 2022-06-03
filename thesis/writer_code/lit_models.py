import itertools
from typing import Optional, Dict, Union, Tuple, Any, List
from pathlib import Path

import numpy as np
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Subset

from htr.metrics import CharacterErrorRate, WordErrorRate
from thesis.lit_callbacks import LogLearnableInnerLoopLearningRates
from thesis.lit_models import LitMAMLLearner, LitBaseEpisodic, LitBaseNonEpisodic
from thesis.writer_code.lit_callbacks import LogWorstPredictions, LogModelPredictions
from thesis.writer_code.models import (
    WriterCodeAdaptiveModel,
    WriterCodeAdaptiveModelMAML,
    WriterCodeAdaptiveModelNonEpisodic,
)
from thesis.util import (
    train_split_batch_for_adaptation,
    PREDICTIONS_TO_LOG,
    TrainMode,
    chunk_batch,
    test_split_batch_for_adaptation,
    set_batchnorm_layers_train,
)

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead
from htr.models.lit_models import LitShowAttendRead, LitFullPageHTREncoderDecoder
from htr.util import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from thesis.writer_code.util import (
    AdaptationMethod,
    ADAPTATION_METHODS,
)


class LitWriterCodeAdaptiveModelMAML(LitMAMLLearner):
    def __init__(self, base_model: nn.Module, code_size: int, **kwargs):
        super().__init__(**kwargs)
        self.model = WriterCodeAdaptiveModelMAML(
            base_model=base_model,
            code_size=code_size,
            **kwargs,
        )
        self.save_hyperparameters("code_size")
        self.save_hyperparameters(self.hparams_to_log)

    def add_model_specific_callbacks(
        self, callbacks: List[Callback], label_encoder: LabelEncoder, **kwargs
    ) -> List[Callback]:
        callbacks = super().add_model_specific_callbacks(
            callbacks,
            label_encoder=label_encoder,
            **kwargs,
        )
        callbacks.append(LogLearnableInnerLoopLearningRates())
        return callbacks

    @staticmethod
    def init_with_base_model_from_checkpoint(**kwargs):
        return LitWriterCodeAdaptiveModel.init_with_base_model_from_checkpoint(**kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitWriterCodeAdaptiveModelMAML")
        parser.add_argument(
            "--code_size",
            type=int,
            default=64,
            help="Size of the writer codes for adaptation.",
        )
        # TODO: some necessary arguments are used from the LitWriterCodeAdaptiveModel
        #  class. Ideally, these are also defined here (but right now would lead to
        #  conflict because both argument parsers are used).
        return parent_parser


class LitWriterCodeAdaptiveModel(LitBaseEpisodic):
    def __init__(
        self,
        base_model: nn.Module,
        cer_metric: CharacterErrorRate,
        wer_metric: WordErrorRate,
        feature_size: int,
        num_writers: int,
        code_size: int = 64,
        adaptation_num_hidden: int = 128,
        ways: int = 8,
        shots: int = 8,
        learning_rate_emb: float = 0.0001,
        weight_decay: float = 0.0001,
        adaptation_opt_steps: int = 1,
        adaptation_method: Union[
            AdaptationMethod, str
        ] = AdaptationMethod.CONDITIONAL_BATCHNORM,
        use_adam_for_adaptation: bool = False,
        max_val_batch_size: int = 128,
        **kwargs,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            cer_metric (CharacterErrorRate): cer metric module
            wer_metric (WordErrorRate): wer metric module
            feature_size (int): size of the feature vectors to adapt, e.g. the output
                feature vectors of a CNN
            num_writers (int): number of writers in the training set
            code_size (int): size of the writer embeddings. If code_size=0, no code
                will be used.
            adaptation_opt_steps (int): number of optimization steps to perform for
                training a new writer code during val/test.
            ways (int): ways
            shots (int): shots
            learning_rate_emb (float): learning rate used for fast adaptation of an
                initial embedding during val/test
            weight_decay (float): weight decay
            adaptation_method (AdaptationMethod): how the writer code should be inserted
                into the model
            use_adam_for_adaptation (bool): whether to use Adam during adaptation
                (otherwise plain SGD is used)
            max_val_batch_size (int): maximum val batch size
        """
        super().__init__(**kwargs)

        assert isinstance(base_model, (FullPageHTREncoderDecoder, ShowAttendRead))

        self.cer_metric = cer_metric
        self.wer_metric = wer_metric
        self.feature_size = feature_size
        self.num_writers = num_writers
        self.code_size = code_size
        self.adaptation_num_hidden = adaptation_num_hidden
        self.ways = ways
        self.shots = shots
        self.learning_rate_emb = learning_rate_emb
        self.weight_decay = weight_decay
        self.adaptation_opt_steps = adaptation_opt_steps
        self.adaptation_method = adaptation_method
        self.use_adam_for_adaptation = use_adam_for_adaptation
        self.max_val_batch_size = max_val_batch_size

        self.ignore_index = base_model.pad_tkn_idx
        self.automatic_optimization = False

        self.model = WriterCodeAdaptiveModel(
            base_model=base_model,
            d_model=feature_size,
            code_size=code_size,
            adaptation_num_hidden=adaptation_num_hidden,
            num_writers=num_writers,
            learning_rate_emb=learning_rate_emb,
            adaptation_opt_steps=adaptation_opt_steps,
            adaptation_method=adaptation_method,
            use_adam_for_adaptation=use_adam_for_adaptation,
            max_val_batch_size=max_val_batch_size,
        )

        self.save_hyperparameters(
            "code_size",
            "adaptation_num_hidden",
            "ways",
            "shots",
            "num_writers",
            "learning_rate_emb",
            "weight_decay",
            "adaptation_opt_steps",
            "adaptation_method",
            "use_adam_for_adaptation",
        )
        self.save_hyperparameters(self.hparams_to_log)

    def forward(
        self,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
        inference_imgs: Tensor,
        inference_tgts: Optional[Tensor] = None,
        mode: TrainMode = TrainMode.TRAIN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self.eval()
        return self.model(
            adaptation_imgs,
            adaptation_targets,
            inference_imgs,
            inference_tgts,
            mode=mode,
        )

    def training_step(self, batch, batch_idx):
        set_batchnorm_layers_train(self.model, False)  # freeze batchnorm stats
        imgs, target, writer_ids = batch
        _, _, loss = self.model(imgs, target, writer_ids, mode=TrainMode.TRAIN)
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
        return self.val_or_test_step(batch, mode=TrainMode.VAL)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.TEST)

    def val_or_test_step(self, batch, mode=TrainMode.VAL):
        """
        Val/test step. The difference with a train step:

        - A new writer code is created based on a small batch of data, named the
          "adaptation batch". This is necessary because the writers in the val/test
          set do not overlap with those in the train set, thus necessitating new
          writer codes.
        - Teacher forcing is not used.
        """
        loss, n_samples = 0, 0
        writerid_to_splits = (
            self.val_writerid_to_splits
            if mode is TrainMode.VAL
            else self.test_writerid_to_splits
        )
        writer_batches = test_split_batch_for_adaptation(
            batch, self.shots, writerid_to_splits
        )

        for adapt_imgs, adapt_tgts, query_imgs, query_tgts in writer_batches:
            if self.code_size == 0:
                # If the writer code is of size 0 (i.e. not used), use the
                # adaptation batch as the only input.
                query_imgs, query_tgts = adapt_imgs, adapt_tgts

            torch.set_grad_enabled(True)
            _, preds, query_loss = self(
                adapt_imgs, adapt_tgts, query_imgs, query_tgts, mode=mode
            )
            torch.set_grad_enabled(False)

            # Log metrics.
            cer_metric = self.model.model.cer_metric
            wer_metric = self.model.model.wer_metric
            cer_metric(preds, query_tgts)
            wer_metric(preds, query_tgts)
            self.log("char_error_rate", cer_metric, prog_bar=False)
            self.log("word_error_rate", wer_metric, prog_bar=True)

            loss += query_loss * query_imgs.size(0)
            n_samples += query_imgs.size(0)
        loss /= n_samples
        self.log(f"{mode.name.lower()}_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # bn_layers = self.model.model.encoder.bn_layers
        # bn_layers_names = set(wn for m in bn_layers for wn, _ in m.named_parameters())
        # encoder_params = self.model.model.encoder.named_parameters()
        # param_group_1 = iter(w for wn, w in encoder_params if not any(wn.endswith(bnn) for bnn in bn_layers_names))
        # param_group_2 = itertools.chain(self.model.writer_code_mlp.parameters(),
        #                                 self.model.writer_embs.parameters(),
        #                                 *(m.parameters() for m in bn_layers))
        # param_groups = [
        #     {"params": param_group_1, "lr": 3e-6},
        #     {"params": param_group_2}
        # ]
        # optimizer = optim.AdamW(
        #     param_groups, lr=self.learning_rate, weight_decay=self.weight_decay
        # )
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.use_cosine_lr_scheduler:
            max_epochs = self.max_epochs or 20
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-06,  # final learning rate
                verbose=True,
            )
            return [optimizer], [lr_scheduler]
        return optimizer

    def add_model_specific_callbacks(
        self,
        callbacks: List[Callback],
        shots: int,
        ways: int,
        label_encoder: LabelEncoder,
        is_train: bool,
    ) -> List[Callback]:
        callbacks = super().add_model_specific_callbacks(
            callbacks,
            shots=shots,
            ways=ways,
            label_encoder=label_encoder,
            is_train=is_train,
        )

        # Prepare fixed batches used for monitoring model predictions during training.
        im, t, wrtrs = next(iter(self.train_dataloader()))
        train_batch = (im[:shots], t[:shots], wrtrs[:shots])
        im, t, wrtrs = next(iter(self.val_dataloader()))
        val_batch = (
            im[:shots],
            t[:shots],
            im[shots : shots + PREDICTIONS_TO_LOG["word"]],
            t[shots : shots + PREDICTIONS_TO_LOG["word"]],
        )
        callbacks.extend(
            [
                LogModelPredictions(
                    label_encoder=label_encoder,
                    val_batch=val_batch,
                    train_batch=train_batch,
                    predict_on_train_start=False,
                ),
                # TODO: add LogWorstPredictions callback.
                # LogWorstPredictions(
                #     label_encoder=label_encoder,
                #     train_dataloader=self.train_dataloader(),
                #     val_dataloader=self.val_dataloader(),
                #     test_dataloader=self.test_dataloader(),
                #     shots=shots,
                #     ways=ways,
                #     training_skipped=not is_train,
                # ),
            ]
        )
        return callbacks

    @staticmethod
    def init_with_base_model_from_checkpoint(
        base_model_arch: str,
        main_model_arch: str,
        checkpoint_path: Union[str, Path],
        model_hparams_file: Union[str, Path],
        label_encoder: LabelEncoder,
        model_params_to_log: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        assert base_model_arch in ["fphtr", "sar"], "Invalid base model architecture."
        assert main_model_arch in [
            "WriterCodeAdaptiveModel",
            "WriterCodeAdaptiveModelMAML",
        ]

        # Initialize base model.
        if base_model_arch == "fphtr":
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
            feature_size = base_model.model.lstm_encoder.rnn_encoder.input_size

        # Initialize meta-model.
        if main_model_arch == "WriterCodeAdaptiveModel":
            model = LitWriterCodeAdaptiveModel.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                cer_metric=base_model.model.cer_metric,
                wer_metric=base_model.model.wer_metric,
                feature_size=feature_size,
                base_model=base_model.model,
                base_model_arch=base_model_arch,
                main_model_arch=main_model_arch,
                **kwargs,
            )
        else:
            model = LitWriterCodeAdaptiveModelMAML.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                cer_metric=base_model.model.cer_metric,
                wer_metric=base_model.model.wer_metric,
                base_model=base_model.model,
                d_model=feature_size,
                base_model_arch=base_model_arch,
                main_model_arch=main_model_arch,
                **kwargs,
            )
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitWriterCodeAdaptiveModel")
        parser.add_argument(
            "--code_size",
            type=int,
            default=1,
            help="Size of the writer embeddings for adaptation.",
        )
        parser.add_argument(
            "--adaptation_num_hidden",
            type=int,
            default=128,
            help="Number of features for the hidden layers of the " "adaptation MLP",
        )
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
            "--adaptation_method",
            type=str,
            default="conditional_batchnorm",
            choices=ADAPTATION_METHODS,
            help="adaptation_method(str): how the writer code should be inserted into the model",
        )
        parser.add_argument(
            "--use_adam_for_adaptation",
            action="store_true",
            default=False,
            help="Use Adam during val/test for training new writer codes.",
        )
        return parent_parser


class LitWriterCodeAdaptiveModelNonEpisodic(LitBaseNonEpisodic):
    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        cer_metric: CharacterErrorRate,
        wer_metric: WordErrorRate,
        code_size: int,
        adaptation_num_hidden: int = 128,
        adaptation_method: Union[
            AdaptationMethod, str
        ] = AdaptationMethod.CONDITIONAL_BATCHNORM,
        **kwargs,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            cer_metric (CharacterErrorRate): cer metric module
            wer_metric (WordErrorRate): wer metric module
            code_size (int): size of the writer codes
            adaptation_num_hidden (int): hidden size for adaptation MLP
            adaptation_method (AdaptationMethod): how the writer code should be inserted
                into the model
        """
        super().__init__(**kwargs)

        assert isinstance(base_model, (FullPageHTREncoderDecoder, ShowAttendRead))

        self.d_model = d_model
        self.cer_metric = cer_metric
        self.wer_metric = wer_metric
        self.code_size = code_size
        self.adaptation_num_hidden = adaptation_num_hidden
        self.adaptation_method = adaptation_method

        self.ignore_index = base_model.pad_tkn_idx
        self.cer_metric = base_model.cer_metric
        self.wer_metric = base_model.wer_metric

        self.model = WriterCodeAdaptiveModelNonEpisodic(
            base_model=base_model,
            d_model=d_model,
            code_size=code_size,
            adaptation_num_hidden=adaptation_num_hidden,
            adaptation_method=adaptation_method,
        )

        self.save_hyperparameters(
            "code_size",
            "adaptation_num_hidden",
            "adaptation_method",
        )
        self.save_hyperparameters(self.hparams_to_log)

    def forward(self, imgs, target, writer_ids, mode) -> Tuple[Tensor, Tensor, Tensor]:
        return self.model(imgs, target, writer_ids, mode=mode)

    def training_step(self, batch, batch_idx):
        set_batchnorm_layers_train(self.model, False)  # freeze batchnorm stats
        imgs, target, writer_ids = batch
        _, _, loss = self.model(imgs, target, writer_ids, mode=TrainMode.TRAIN)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.VAL)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.TEST)

    def val_or_test_step(self, batch, mode=TrainMode.VAL):
        imgs, target, writer_ids = batch
        _, preds, loss = self.model(imgs, target, writer_ids, mode=mode)

        # Log metrics.
        self.cer_metric(preds, target)
        self.wer_metric(preds, target)
        self.log("char_error_rate", self.cer_metric, prog_bar=False)
        self.log("word_error_rate", self.wer_metric, prog_bar=True)
        self.log(f"{mode.name.lower()}_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def add_model_specific_callbacks(
        self,
        callbacks: List[Callback],
        label_encoder: LabelEncoder,
        is_train: bool,
    ) -> List[Callback]:
        # TODO: add LogModelPredictions and LogWorstPredictions callbacks.
        return callbacks

    @staticmethod
    def init_with_base_model_from_checkpoint(
        base_model_arch: str,
        main_model_arch: str,
        checkpoint_path: Union[str, Path],
        model_hparams_file: Union[str, Path],
        label_encoder: LabelEncoder,
        model_params_to_log: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        assert base_model_arch in ["fphtr", "sar"], "Invalid base model architecture."

        # Initialize base model.
        if base_model_arch == "fphtr":
            # Load FPHTR model.
            base_model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
            )
            d_model = base_model.encoder.resnet_out_features
        else:  # SAR
            base_model = LitShowAttendRead.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
            )
            d_model = base_model.model.lstm_encoder.rnn_encoder.input_size

        # Initialize meta-model.
        model = LitWriterCodeAdaptiveModelNonEpisodic.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            cer_metric=base_model.model.cer_metric,
            wer_metric=base_model.model.wer_metric,
            base_model=base_model.model,
            d_model=d_model,
            base_model_arch=base_model_arch,
            main_model_arch=main_model_arch,
            **kwargs,
        )
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitWriterCodeAdaptiveModel")
        parser.add_argument(
            "--adaptation_num_hidden",
            type=int,
            default=128,
            help="Number of features for the hidden layers of the MLP used for adaptation",
        )
        parser.add_argument(
            "--adaptation_method",
            type=str,
            default="conditional_batchnorm",
            choices=ADAPTATION_METHODS,
            help="adaptation_method(str): how the writer code should be inserted into the model",
        )
        return parent_parser
