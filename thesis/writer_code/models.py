from functools import partial
from typing import Optional, Callable, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
from torch import Tensor

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead

from thesis.writer_code.util import WriterEmbeddingType
from thesis.models import MAMLLearner
from thesis.util import (
    freeze,
    TrainMode,
    split_batch_for_adaptation,
    set_dropout_layers_train,
    set_batchnorm_layers_train,
)


class WriterCode(nn.Module):
    """A simple Module wrapper for a writer code."""

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.writer_code = torch.empty(1, num_features)
        self.writer_code.normal_()  # mean 0, std 1
        self.writer_code = nn.Parameter(self.writer_code)

    def forward(self, *args, **kwargs):
        return self.writer_code


class WriterCodeAdaptiveModelMAML(nn.Module, MAMLLearner):
    """<Model description here.>"""  # TODO

    meta_weights = ["gbml.module.writer_code", "adaptation"]  # TODO: verify

    def __init__(
        self,
        base_model: nn.Module,
        base_model_arch: str,
        d_model: int,
        emb_size: int,
        adaptation_num_hidden: int,
        num_writers: int,
        ways: int = 8,
        shots: int = 16,
        val_batch_size: int = 64,
        use_batch_stats_for_batchnorm: bool = False,
        use_dropout: bool = False,
        num_inner_steps: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.base_model_arch = base_model_arch
        self.d_model = d_model
        self.emb_size = emb_size
        self.adaptation_num_hidden = adaptation_num_hidden
        self.num_writers = num_writers
        self.ways = ways
        self.shots = shots
        self.val_batch_size = val_batch_size
        self.use_batch_stats_for_batchnorm = use_batch_stats_for_batchnorm
        self.use_dropout = use_dropout
        self.num_inner_steps = num_inner_steps

        assert base_model.loss_fn.reduction == "mean"

        self.gbml = l2l.algorithms.MetaSGD(WriterCode(emb_size), first_order=False)
        self.adaptation = FeatureTransform(
            AdaptationMLP(d_model, emb_size, adaptation_num_hidden)
        )
        self.base_model_with_adaptation = BaseModelAdaptation(base_model)

        freeze(base_model)  # make sure the base model weights are frozen
        # Finetune the linear layer in the base model directly following the adaptation
        # model.
        base_model.encoder.linear.requires_grad_(True)

    def forward(
        self, imgs: Tensor, target: Tensor, writer_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        intermediate_transform = partial(
            self.adaptation, writer_emb=self.gbml.module.writer_code
        )
        logits, loss = self.base_model_with_adaptation(
            imgs,
            target,
            intermediate_transform=intermediate_transform,
            teacher_forcing=False,
        )
        sampled_ids = logits.argmax(-1)
        return logits, sampled_ids, loss

    def meta_learn(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.TRAIN
    ) -> Tuple[Tensor, float, Optional[Dict[int, List]]]:
        outer_loss = 0.0
        inner_losses = []
        imgs, target, writer_ids = batch

        assert imgs.size(0) >= 2 * self.ways * self.shots, imgs.size(0)

        # Split the batch into N different writers, for K-shot adaptation.
        tasks = split_batch_for_adaptation(
            batch, self.ways, self.shots, limit_num_samples_per_task=self.val_batch_size
        )

        for support_imgs, support_tgts, query_imgs, query_tgts in tasks:
            # Calling `model.clone()` allows updating the module while still allowing
            # computation of derivatives of the new modules' parameters w.r.t. the
            # original parameters.
            learner = self.gbml.clone()

            # Inner loop.
            assert torch.is_grad_enabled()
            for _ in range(self.num_inner_steps):
                # Adapt the model to the support data.
                learner, support_loss, _ = self.fast_adaptation(
                    learner, support_imgs, support_tgts
                )
                inner_losses.append(support_loss.item())

            # Outer loop.
            intermediate_transform = partial(
                self.adaptation, writer_emb=learner.module.writer_code
            )
            if mode is TrainMode.TRAIN:
                set_dropout_layers_train(self, self.use_dropout)
                _, query_loss = self.base_model_with_adaptation(
                    query_imgs,
                    query_tgts,
                    intermediate_transform=intermediate_transform,
                    teacher_forcing=True,
                )
                # Using the torch `backward()` function rather than PLs
                # `manual_backward` means that mixed precision cannot be used.
                (query_loss / self.ways).backward()
            else:  # val/test
                with torch.inference_mode():
                    logits, query_loss = self.base_model_with_adaptation(
                        query_imgs,
                        query_tgts,
                        intermediate_transform=intermediate_transform,
                        teacher_forcing=False,
                    )
                    preds = logits.argmax(-1)

                # Calculate metrics.
                self.base_model_with_adaptation.model.cer_metric(preds, query_tgts)
                self.base_model_with_adaptation.model.wer_metric(preds, query_tgts)
            outer_loss += query_loss

        outer_loss /= self.ways
        inner_loss_avg = np.mean(inner_losses)

        return outer_loss, inner_loss_avg, None

    def fast_adaptation(
        self,
        learner: l2l.algorithms.MetaSGD,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
    ) -> Tuple[Any, float, Optional[Tensor]]:
        """Takes a single gradient step on a batch of data."""
        set_dropout_layers_train(self, False)  # disable dropout
        set_batchnorm_layers_train(self, self.use_batch_stats_for_batchnorm)

        intermediate_transform = partial(
            self.adaptation, writer_emb=learner.module.writer_code
        )
        _, support_loss = self.base_model_with_adaptation(
            adaptation_imgs,
            adaptation_targets,
            intermediate_transform=intermediate_transform,
            teacher_forcing=True,
        )

        # Calculate gradients and take an optimization step.
        learner.adapt(support_loss)

        return learner, support_loss, None


class WriterCodeAdaptiveModel(nn.Module):
    """
    Implementation of speaker-adaptive model by Abdel-Hamid et al. (2013),
    adapted to HTR models.
    """

    meta_weights = ["writer_embs", "adaptation"]

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        emb_size: int,
        adaptation_num_hidden: int,
        num_writers: int,
        learning_rate_emb: float,
        embedding_type: WriterEmbeddingType = WriterEmbeddingType.LEARNED,
        adaptation_opt_steps: int = 1,
        use_adam_for_adaptation: bool = False,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            emb_size (int): size of the writer embeddings
            adaptation_num_hidden (int): hidden size for adaptation MLP
            num_writers (int): number of writers in the training set
            learning_rate_emb (float): learning rate used for fast adaptation of an
                initial embedding during val/test
            embedding_type (WriterEmbeddingType): type of writer embedding used
            adaptation_opt_steps (int): number of optimization steps during adaptation
            use_adam_for_adaptation (bool): whether to use Adam during adaptation
                (otherwise plain SGD is used)
        """
        super().__init__()
        self.d_model = d_model
        self.emb_size = emb_size
        self.adaptation_num_hidden = adaptation_num_hidden
        self.num_writers = num_writers
        self.learning_rate_emb = learning_rate_emb
        self.embedding_type = embedding_type
        self.adaptation_opt_steps = adaptation_opt_steps
        self.use_adam_for_adaptation = use_adam_for_adaptation

        self.emb_transform = None
        if self.embedding_type == WriterEmbeddingType.LEARNED:
            self.writer_embs = nn.Embedding(num_writers, emb_size)
        if self.embedding_type == WriterEmbeddingType.TRANSFORMED:
            # self.emb_transform = nn.Linear(d_model, emb_size)
            self.emb_transform = nn.LSTM(
                # These settings were chosen ad-hoc.
                input_size=d_model,
                hidden_size=emb_size,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=False,
            )
        self.feature_transform = FeatureTransform(
            AdaptationMLP(d_model, emb_size, adaptation_num_hidden), self.emb_transform
        )
        self.base_model_with_adaptation = BaseModelAdaptation(base_model)

        assert base_model.loss_fn.reduction == "mean"

        freeze(base_model)  # make sure the base model weights are frozen
        # Finetune the linear layer in the base model directly following the adaptation
        # model.
        base_model.encoder.linear.requires_grad_(True)

    def forward(
        self, *args, mode: str = "train", **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert mode in ["train", "val", "test"]
        if self.embedding_type == WriterEmbeddingType.LEARNED:
            if mode == "train":  # use a pre-trained embedding
                logits, loss = self.forward_existing_code(*args, **kwargs)
            else:  # initialize and train a new writer embedding
                logits, loss = self.forward_new_code(*args, **kwargs)
        else:  # WriterEmbeddingType.TRANSFORMED
            logits, loss = self.forward_transform(*args, **kwargs)
        sampled_ids = logits.argmax(-1)
        return logits, sampled_ids, loss

    def forward_transform(
        self, imgs: Tensor, target: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Create a writer code by passing the average CNN feature vector through a MLP.
        """
        logits, loss = self.base_model_with_adaptation(
            imgs,
            target,
            intermediate_transform=self.feature_transform,
            teacher_forcing=True,
        )
        return logits, loss

    def forward_existing_code(
        self, imgs: Tensor, target: Tensor, writer_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Perform adaptation using an existing writer code."""
        # Load the writer code.
        writer_emb = self.writer_embs(writer_ids)  # (N, emb_size)

        # Run inference using the writer code.
        intermediate_transform = partial(self.feature_transform, writer_emb=writer_emb)
        logits, loss = self.base_model_with_adaptation(
            imgs,
            target,
            intermediate_transform=intermediate_transform,
            teacher_forcing=True,
        )
        return logits, loss

    def forward_new_code(
        self,
        adapt_imgs: Tensor,
        adapt_tgts: Tensor,
        inference_imgs: Tensor,
        inference_tgts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Create a writer embedding based on a set of adaptation images, and run
        inference on another set.

        Args:
            adapt_imgs (Tensor): images to do adaptation on
            adapt_tgts (Tensor): targets for `adaptation_imgs`
            inference_imgs (Tensor): images to make predictions on
            inference_tgts (Optional[Tensor]): targets for `inference_imgs`
        """
        # Train a writer code.
        writer_code = self.new_writer_code(adapt_imgs, adapt_tgts)

        # Run inference using the writer code.
        intermediate_transform = partial(self.feature_transform, writer_emb=writer_code)
        with torch.inference_mode():
            logits, loss = self.base_model_with_adaptation(
                inference_imgs,
                inference_tgts,
                intermediate_transform=intermediate_transform,
                teacher_forcing=False,
            )
        return logits, loss

    def new_writer_code(
        self, adaptation_imgs: Tensor, adaptation_targets: Tensor
    ) -> Tensor:
        """
        Create a new writer code (embedding) based on a batch of examples for a writer.

        The writer code is created by running one or multiple forward/backward
        passes on a batch of adaptation data in order to train a new writer
        embedding.
        """
        writer_emb = torch.empty(1, self.emb_size, device=adaptation_imgs.device)
        writer_emb.normal_()  # mean 0, std 1
        writer_emb.requires_grad = True

        if self.use_adam_for_adaptation:
            optimizer = optim.Adam(iter([writer_emb]), lr=self.learning_rate_emb)
        else:
            # Plain SGD, i.e. update rule `g = g - lr * grad(loss)`
            optimizer = optim.SGD(iter([writer_emb]), lr=self.learning_rate_emb)

        for _ in range(self.adaptation_opt_steps):
            intermediate_transform = partial(
                self.feature_transform, writer_emb=writer_emb
            )
            _, loss = self.base_model_with_adaptation(
                adaptation_imgs,
                adaptation_targets,
                intermediate_transform=intermediate_transform,
                teacher_forcing=False,
            )

            optimizer.zero_grad()
            loss.backward(inputs=writer_emb)
            optimizer.step()  # update the embedding
        writer_emb.detach_()

        return writer_emb


class AdaptationMLP(nn.Module):
    def __init__(self, d_model: int, emb_size: int, num_hidden: int):
        """
        A multi-layer perceptron used for adapting features based on writer embeddings.

        Args:
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            emb_size (int): size of the writer embeddings
            num_hidden (int): hidden size for adaptation MLP
        """
        super().__init__()

        self.d_model = d_model
        self.emb_size = emb_size
        self.num_hidden = num_hidden

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model + emb_size, num_hidden),
                    nn.ReLU(inplace=True),
                    # BatchNorm1dPermute(num_hidden),
                ),
                nn.Sequential(
                    nn.Linear(num_hidden + emb_size, num_hidden),
                    nn.ReLU(inplace=True),
                    # BatchNorm1dPermute(num_hidden),
                ),
                nn.Sequential(nn.Linear(num_hidden + emb_size, d_model)),
            ]
        )

    def forward(self, features: Tensor, writer_emb: Tensor) -> Tensor:
        """
        Adapt features based on a writer code (embedding).

        Features are adapted by passing them through an adaptation network,
        along with a writer embedding. The writer embedding is added as
        additional input for each layer of the adaptation network by concatenating
        the intermediate feature vector and the writer embedding.

        Args:
             features (Tensor of shape (N, d_model, *)): features to adapt
             writer_emb (Tensor of shape (N, emb_size)): writer embeddings, used for adapting the features
        Returns:
            Tensor of shape (N, d_model, *), containing transformed features
        """
        features = features.movedim(1, -1)  # (N, *, d_model)
        extra_dims = features.ndim - writer_emb.ndim
        for _ in range(extra_dims):
            writer_emb = writer_emb.unsqueeze(1)
        writer_emb = writer_emb.expand(*features.shape[:-1], -1)
        res = features
        for layer in self.layers:
            res = torch.cat((res, writer_emb), -1)
            res = layer(res)
        return (res + features).movedim(-1, 1)


class FeatureTransform(nn.Module):
    def __init__(
        self,
        adaptation_transform: nn.Module,
        emb_transform: Optional[nn.Module] = None,
    ):
        """
        Args:
            adaptation_transform (nn.Module): transformation from features and
                writer embedding to adapted features
            emb_transform (Optional[nn.Module]): transformation from features to a
                writer embedding.
        """
        super().__init__()
        self.emb_transform = emb_transform
        self.adaptation_transform = adaptation_transform

    def forward(self, features: Tensor, writer_emb: Optional[Tensor] = None):
        """
        Transform features using a writer code.

        Args:
            features (Tensor of shape (N, d_model, *)): features to adapt
            writer_emb (Optional[Tensor] of shape (N, emb_size): writer embedding to be
                used for adaptation. If not specified, a learnable feature transformation
                is used to create the embedding.
        Returns:
            Tensor of shape (N, d_model, *) containing adapted features
        """
        if writer_emb is None:
            # Transform features to create writer embedding.
            # Average across spatial dimensions.
            # h_feat = features.size(2)
            # feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
            # feat_v = feat_v.squeeze(2)  # bsz * C * W
            feats_lstm = features.flatten(2, -1).permute(0, 2, 1).contiguous()
            writer_emb = self.emb_transform(feats_lstm)[0][:, -1, :]
        return self.adaptation_transform(features, writer_emb)


class BaseModelAdaptation(nn.Module):
    """
    Base HTR model with an adaptation transformation injected in between.

    Specifically, the adaptation model is injected right after the CNN feature map
    output.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        if isinstance(model, FullPageHTREncoderDecoder):
            self.arch = "fphtr"
        elif isinstance(model, ShowAttendRead):
            self.arch = "sar"
        else:
            raise ValueError(f"Unrecognized model class: {model.__class__}")

    def forward(
        self,
        imgs: Tensor,
        target: Optional[Tensor] = None,
        intermediate_transform: Optional[Callable] = None,
        teacher_forcing: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass using an intermediate transform."""
        if self.arch == "fphtr":
            features = self.model.encoder(
                imgs, intermediate_transform=intermediate_transform
            )
            if teacher_forcing:
                logits = self.model.decoder.decode_teacher_forcing(features, target)
            else:
                logits, _ = self.model.decoder(features)
        else:  # SAR
            features = self.model.resnet_encoder(imgs)
            features = intermediate_transform(features)
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


class BatchNorm1dPermute(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor of shape (N, *, k)
        """
        x = x.movedim(-1, 1)  # (N, k, *)
        x = self.bn(x)
        x = x.movedim(1, -1)  # (N, *, k)
        return x
