from functools import partial
import math
from typing import Optional, Callable, Tuple, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import learn2learn as l2l
from torch import Tensor

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead

from thesis.writer_code.util import WriterEmbeddingType, AdaptationMethod
from thesis.models import MAMLLearner
from thesis.util import (
    freeze,
    TrainMode,
    split_batch_for_adaptation,
    set_dropout_layers_train,
    set_batchnorm_layers_train,
    chunk_batch,
    collect_bn_layers,
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

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        code_size: int,
        adaptation_num_hidden: int,
        ways: int = 8,
        shots: int = 16,
        max_val_batch_size: int = 64,
        use_batch_stats_for_batchnorm: bool = False,
        use_dropout: bool = False,
        num_inner_steps: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.code_size = code_size
        self.adaptation_num_hidden = adaptation_num_hidden
        self.ways = ways
        self.shots = shots
        self.max_val_batch_size = max_val_batch_size
        self.use_batch_stats_for_batchnorm = use_batch_stats_for_batchnorm
        self.use_dropout = use_dropout
        self.num_inner_steps = num_inner_steps

        assert base_model.loss_fn.reduction == "mean"

        self.gbml = l2l.algorithms.MetaSGD(WriterCode(code_size), first_order=False)
        self.adaptation = FeatureTransform(
            AdaptationMLP(d_model, code_size, adaptation_num_hidden)
        )
        self.base_model_with_adaptation = BaseModelAdaptation(base_model)

        freeze(base_model)  # make sure the base model weights are frozen
        # Finetune the linear layer in the base model directly following the adaptation
        # model.
        base_model.encoder.linear.requires_grad_(True)

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
        learner = self.gbml.clone()

        # Adapt the model.
        for _ in range(self.num_inner_steps):
            learner, support_loss, _ = self.fast_adaptation(
                learner, adaptation_imgs, adaptation_targets
            )

        # Run inference on the adapted model.
        intermediate_transform = partial(
            self.adaptation, writer_code=learner.module.writer_code
        )
        with torch.inference_mode():
            logits, _ = self.base_model_with_adaptation(
                inference_imgs,
                intermediate_transform=intermediate_transform,
                teacher_forcing=False,
            )
            sampled_ids = logits.argmax(-1)

        return logits, sampled_ids

    def meta_learn(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.TRAIN
    ) -> Tuple[Tensor, float, Optional[Dict[int, List]]]:
        outer_loss, n_query_images = 0.0, 0
        inner_losses = []
        imgs, target, writer_ids = batch

        assert imgs.size(0) >= 2 * self.ways * self.shots, imgs.size(0)

        # Split the batch into N different writers, for K-shot adaptation.
        tasks = split_batch_for_adaptation(batch, self.ways, self.shots)

        for support_imgs, support_tgts, query_imgs, query_tgts in tasks:
            n_query_images += query_imgs.size(0)

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
                self.adaptation, writer_code=learner.module.writer_code
            )
            if mode is TrainMode.TRAIN:
                set_dropout_layers_train(learner, self.use_dropout)
                _, query_loss = self.base_model_with_adaptation(
                    query_imgs,
                    query_tgts,
                    intermediate_transform=intermediate_transform,
                    teacher_forcing=True,
                )
                # Using the torch `backward()` function rather than PLs
                # `manual_backward` means that mixed precision cannot be used.
                (query_loss / self.ways).backward()
                outer_loss += query_loss
            else:  # val/test
                # The set of writer examples may be too large too fit into a single
                # batch. Therefore, chunk the data and process each chunk individually.
                query_img_chunks, query_tgt_chunks = chunk_batch(
                    query_imgs, query_tgts, self.max_val_batch_size
                )

                for img, tgt in zip(query_img_chunks, query_tgt_chunks):
                    with torch.inference_mode():
                        logits, query_loss = self.base_model_with_adaptation(
                            img,
                            tgt,
                            intermediate_transform=intermediate_transform,
                            teacher_forcing=False,
                        )
                        preds = logits.argmax(-1)

                    # Calculate metrics.
                    self.base_model_with_adaptation.model.cer_metric(preds, tgt)
                    self.base_model_with_adaptation.model.wer_metric(preds, tgt)
                    outer_loss += query_loss * img.size(0)

        outer_loss /= n_query_images
        inner_loss_avg = np.mean(inner_losses)

        return outer_loss, inner_loss_avg, None

    def fast_adaptation(
        self,
        learner: l2l.algorithms.MetaSGD,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
    ) -> Tuple[Any, float, Optional[Tensor]]:
        """Takes a single gradient step on a batch of data."""
        set_dropout_layers_train(learner, False)  # disable dropout
        set_batchnorm_layers_train(learner, self.use_batch_stats_for_batchnorm)

        intermediate_transform = partial(
            self.adaptation, writer_code=learner.module.writer_code
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

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        code_size: int,
        adaptation_num_hidden: int,
        num_writers: int,
        learning_rate_emb: float,
        ways: int = 8,
        shots: int = 8,
        max_val_batch_size: int = 64,
        embedding_type: WriterEmbeddingType = WriterEmbeddingType.LEARNED,
        adaptation_opt_steps: int = 1,
        use_adam_for_adaptation: bool = False,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            code_size (int): size of the writer embeddings. If code_size=0, no code
                will be used.
            adaptation_num_hidden (int): hidden size for adaptation MLP
            num_writers (int): number of writers in the training set
            learning_rate_emb (float): learning rate used for fast adaptation of an
                initial embedding during val/test
            ways (int): ways
            shots (int): shots
            max_val_batch_size (int): maximum val batch size
            embedding_type (WriterEmbeddingType): type of writer embedding used
            adaptation_opt_steps (int): number of optimization steps during adaptation
            use_adam_for_adaptation (bool): whether to use Adam during adaptation
                (otherwise plain SGD is used)
        """
        super().__init__()
        self.d_model = d_model
        self.code_size = code_size
        self.adaptation_num_hidden = adaptation_num_hidden
        self.num_writers = num_writers
        self.learning_rate_emb = learning_rate_emb
        self.ways = ways
        self.shots = shots
        self.max_val_batch_size = max_val_batch_size
        self.embedding_type = embedding_type
        self.adaptation_opt_steps = adaptation_opt_steps
        self.use_adam_for_adaptation = use_adam_for_adaptation

        self.emb_transform = None
        self.writer_embs = None
        if self.embedding_type == WriterEmbeddingType.LEARNED and code_size > 0:
            self.writer_embs = nn.Embedding(num_writers, code_size)
        if self.embedding_type == WriterEmbeddingType.TRANSFORMED:
            # self.emb_transform = nn.Linear(d_model, code_size)
            self.emb_transform = nn.LSTM(
                # These settings were chosen ad-hoc.
                input_size=d_model,
                hidden_size=code_size,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=False,
            )
        self.feature_transform = FeatureTransform(
            AdaptationMLP(d_model, code_size, adaptation_num_hidden),
            self.emb_transform,
            generate_code=(embedding_type == WriterEmbeddingType.TRANSFORMED),
        )
        self.base_model_with_adaptation = BaseModelAdaptation(base_model)

        assert base_model.loss_fn.reduction == "mean"

        freeze(base_model)  # make sure the base model weights are frozen
        # Finetune the linear layer in the base model directly following the adaptation
        # model.
        base_model.encoder.linear.requires_grad_(True)

    def forward(
        self, *args, mode: TrainMode = TrainMode.TRAIN, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.embedding_type == WriterEmbeddingType.LEARNED:
            if mode == TrainMode.TRAIN:
                # Use a pre-trained embedding.
                logits, loss = self.forward_existing_code(*args, **kwargs)
            else:  # initialize and train a new writer embedding
                logits, loss = self.forward_new_code(*args, **kwargs)
        elif self.embedding_type == WriterEmbeddingType.TRANSFORMED:
            logits, loss = self.forward_transform(*args, **kwargs)
        else:
            raise ValueError(f"Unrecognized emb type: {mode}")
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
        self, imgs: Tensor, target: Tensor, writer_ids: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Perform adaptation using an existing writer code."""
        # Load the writer code.
        writer_code = None
        if self.code_size > 0:
            writer_code = self.writer_embs(writer_ids)  # (N, code_size)

        # Run inference using the writer code.
        intermediate_transform = partial(
            self.feature_transform, writer_code=writer_code
        )
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
        Create a writer code based on a set of adaptation images, and run
        inference on another set.

        Args:
            adapt_imgs (Tensor): images to do adaptation on
            adapt_tgts (Tensor): targets for `adaptation_imgs`
            inference_imgs (Tensor): images to make predictions on
            inference_tgts (Optional[Tensor]): targets for `inference_imgs`
        """
        # Train a writer code.
        writer_code = None
        if self.code_size > 0:
            writer_code = self.new_writer_code(adapt_imgs, adapt_tgts)
        else:
            # If the writer code is of size 0 (i.e. not used), use the
            # adaptation batch as the only input.
            inference_imgs, inference_tgts = adapt_imgs, adapt_tgts

        # The set of writer examples may be too large too fit into a single
        # batch. Therefore, chunk the data and process each chunk individually.
        # TODO: this does not work if inference_tgts=None
        inf_img_chunks, inf_tgt_chunks = chunk_batch(
            inference_imgs, inference_tgts, self.max_val_batch_size
        )

        # Run inference using the writer code.
        inference_loss = 0.0
        all_logits = []
        intermediate_transform = partial(
            self.feature_transform, writer_code=writer_code
        )
        for img, tgt in zip(inf_img_chunks, inf_tgt_chunks):
            with torch.inference_mode():
                logits, loss = self.base_model_with_adaptation(
                    img,
                    tgt,
                    intermediate_transform=intermediate_transform,
                    teacher_forcing=False,
                )
            all_logits.append(logits)
            inference_loss += loss * img.size(0)
        inference_loss /= inference_imgs.size(0)
        max_seq_len = max(t.size(1) for t in all_logits)
        logits = torch.cat(
            [F.pad(t, (0, 0, 0, max_seq_len - t.size(1))) for t in all_logits], 0
        )
        # TODO: is it okay to pad logits with zeros?
        return logits, inference_loss

    def new_writer_code(
        self, adaptation_imgs: Tensor, adaptation_targets: Tensor
    ) -> Tensor:
        """
        Create a new writer code (embedding) based on a batch of examples for a writer.

        The writer code is created by running one or multiple forward/backward
        passes on a batch of adaptation data in order to train a new writer
        embedding.
        """
        writer_code = torch.empty(1, self.code_size, device=adaptation_imgs.device)
        writer_code.normal_()  # mean 0, std 1
        writer_code.requires_grad = True

        if self.use_adam_for_adaptation:
            optimizer = optim.Adam(iter([writer_code]), lr=self.learning_rate_emb)
        else:
            # Plain SGD, i.e. update rule `g = g - lr * grad(loss)`
            optimizer = optim.SGD(iter([writer_code]), lr=self.learning_rate_emb)

        for _ in range(self.adaptation_opt_steps):
            intermediate_transform = partial(
                self.feature_transform, writer_code=writer_code
            )
            _, loss = self.base_model_with_adaptation(
                adaptation_imgs,
                adaptation_targets,
                intermediate_transform=intermediate_transform,
                teacher_forcing=False,
            )

            optimizer.zero_grad()
            loss.backward(inputs=writer_code)
            optimizer.step()  # update the embedding
        writer_code.detach_()

        return writer_code


class WriterCodeAdaptiveModelNonEpisodic(nn.Module):
    """
    Writer-code based adaptation without episodic training. Effectively, this means the
    writer codes are loaded into the model in advance and do not require training.
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        writer_codes: Dict[str, np.array],
        code_size: int,
        code_name: str,
        adaptation_num_hidden: int,
        adaptation_method: Union[
            AdaptationMethod, str
        ] = AdaptationMethod.CONDITIONAL_BATCHNORM,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            writer_codes (Dict[str, np.array)): dictionary mapping writer identity to writer code
            code_size (int): size of the writer embeddings. If code_size=0, no code
                will be used.
            code_name (str): type of code to use, e.g. Hinge
            adaptation_num_hidden (int): hidden size for adaptation MLP
            adaptation_method (AdaptationMethod): how the writer code should be inserted into the model
        """
        super().__init__()
        self.d_model = d_model
        self.writer_codes = writer_codes
        self.code_size = code_size
        self.code_name = code_name
        self.adaptation_num_hidden = adaptation_num_hidden
        self.adaptation_method = adaptation_method
        if isinstance(self.adaptation_method, str):
            self.adaptation_method = AdaptationMethod.from_string(
                self.adaptation_method
            )

        if isinstance(base_model, FullPageHTREncoderDecoder):
            self.arch = "fphtr"
        elif isinstance(base_model, ShowAttendRead):
            self.arch = "sar"
        else:
            raise ValueError(f"Unrecognized model class: {base_model.__class__}")

        assert base_model.loss_fn.reduction == "mean"

        freeze(base_model)  # make sure the base model weights are frozen
        # Finetune the linear layer in the base model directly following the adaptation
        # model.
        # base_model.encoder.linear.requires_grad_(True)

        self.feature_transform = None
        if self.adaptation_method is AdaptationMethod.CONDITIONAL_BATCHNORM:
            # resnet_old = (
            #     base_model.encoder
            #     if self.arch == "fphtr"
            #     else base_model.resnet_encoder
            # )
            # resnet_new = WriterAdaptiveResnet(
            #     resnet_old, code_size, adaptation_num_hidden
            # )
            # if self.arch == "fphtr":
            #     base_model.encoder = resnet_new
            # else:  # SAR
            #     base_model.resnet_encoder = resnet_new
            self.model = base_model
        elif self.adaptation_method is AdaptationMethod.CNN_OUTPUT:
            self.feature_transform = FeatureTransform(
                AdaptationMLP(d_model, code_size, adaptation_num_hidden),
                generate_code=False,
            )
            base_model.encoder.linear.requires_grad_(True)
            self.model = BaseModelAdaptation(base_model)
        else:
            raise ValueError(f"Incorrect adaptation method: {self.adaptation_method}")

    def forward(
        self,
        imgs: Tensor,
        target: Tensor,
        writer_ids: Tensor,
        mode: TrainMode = TrainMode.TRAIN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        writer_code = torch.from_numpy(
            np.stack([self.writer_codes[writer.item()] for writer in writer_ids], 0)
        ).to(
            imgs.device
        )  # writer_code: (N, code_size)

        teacher_forcing = True if mode == TrainMode.TRAIN else False
        if self.adaptation_method is AdaptationMethod.CNN_OUTPUT:
            intermediate_transform = partial(
                self.feature_transform, writer_code=writer_code
            )
            logits, loss = self.model(
                imgs,
                target,
                intermediate_transform=intermediate_transform,
                teacher_forcing=teacher_forcing,
            )
        else:
            if self.arch == "fphtr":
                features = self.model.encoder(imgs, writer_code)
                if teacher_forcing:
                    logits = self.model.decoder.decode_teacher_forcing(features, target)
                else:
                    logits, _ = self.model.decoder(features)
            else:  # SAR
                imgs = imgs.unsqueeze(1)
                features = self.model.resnet_encoder(imgs, writer_code)
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
        sampled_ids = logits.argmax(-1)
        return logits, sampled_ids, loss


class AdaptationMLP(nn.Module):
    def __init__(self, d_model: int, code_size: int, num_hidden: int):
        """
        A multi-layer perceptron used for adapting features based on writer codes.

        Args:
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            code_size (int): size of the writer codes
            num_hidden (int): hidden size for adaptation MLP
        """
        super().__init__()

        self.d_model = d_model
        self.code_size = code_size
        self.num_hidden = num_hidden

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model + code_size, num_hidden),
                    nn.ReLU(inplace=True),
                    # BatchNorm1dPermute(num_hidden),
                ),
                nn.Sequential(
                    nn.Linear(num_hidden + code_size, num_hidden),
                    nn.ReLU(inplace=True),
                    # BatchNorm1dPermute(num_hidden),
                ),
                nn.Sequential(nn.Linear(num_hidden + code_size, d_model)),
            ]
        )

    def forward(self, features: Tensor, writer_code: Optional[Tensor] = None) -> Tensor:
        """
        Adapt features based on a writer code (embedding).

        Features are adapted by passing them through an adaptation network,
        along with a writer embedding. The writer embedding is added as
        additional input for each layer of the adaptation network by concatenating
        the intermediate feature vector and the writer embedding.

        Args:
             features (Tensor of shape (N, d_model, *)): features to adapt
             writer_code (Optional tensor of shape (N, code_size)): writer codes,
                used for adapting the features
        Returns:
            Tensor of shape (N, d_model, *), containing transformed features
        """
        features = features.movedim(1, -1)  # (N, *, d_model)
        if writer_code is not None:
            extra_dims = features.ndim - writer_code.ndim
            for _ in range(extra_dims):
                writer_code = writer_code.unsqueeze(1)
            writer_code = writer_code.expand(*features.shape[:-1], -1)
        res = features
        for layer in self.layers:
            if writer_code is not None:
                res = torch.cat((res, writer_code), -1)
            res = layer(res)
        return (res + features).movedim(-1, 1)


class FeatureTransform(nn.Module):
    def __init__(
        self,
        adaptation_transform: nn.Module,
        emb_transform: Optional[nn.Module] = None,
        generate_code: bool = False,
    ):
        """
        Args:
            adaptation_transform (nn.Module): transformation from features and
                writer embedding to adapted features
            emb_transform (Optional[nn.Module]): transformation from features to a
                writer embedding.
            generate_code (bool): whether to use a learned transform to create a
                write code.
        """
        super().__init__()
        self.emb_transform = emb_transform
        self.adaptation_transform = adaptation_transform
        self.generate_code = generate_code

    def forward(self, features: Tensor, writer_code: Optional[Tensor] = None):
        """
        Transform features using a writer code.

        Args:
            features (Tensor of shape (N, d_model, *)): features to adapt
            writer_code (Optional[Tensor] of shape (N, code_size): writer embedding to be
                used for adaptation. If not specified, a learnable feature transformation
                is used to create the embedding.
        Returns:
            Tensor of shape (N, d_model, *) containing adapted features
        """
        if writer_code is None and self.generate_code:
            # Transform features to create writer embedding.
            # Average across spatial dimensions.
            # h_feat = features.size(2)
            # feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
            # feat_v = feat_v.squeeze(2)  # bsz * C * W
            feats_lstm = features.flatten(2, -1).permute(0, 2, 1).contiguous()
            writer_code = self.emb_transform(feats_lstm)[0][:, -1, :]
        return self.adaptation_transform(features, writer_code)


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


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional batch normalization. Predict deltas to the batchnorm affine
    parameters by linear transform of a writer code.

    The affine delta parameters for the batchnorm weight and bias are obtained by
    feeding a writer code through a linear mapping producing 2 * C outputs, where C
    is the number of output channels.
    """

    def __init__(
        self,
        batchnorm_layer: nn.BatchNorm2d,
        writer_code_size: int,
        adaptation_num_hidden: int = 128,
    ):
        super().__init__()
        self.bn = batchnorm_layer
        self.writer_code_size = writer_code_size
        self.writer_code = None
        self.adapt = nn.Sequential(  # 1-hidden-layer MLP
            nn.Linear(writer_code_size, adaptation_num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(adaptation_num_hidden, 2 * batchnorm_layer.num_features),
        )

        # Save batchnorm affine parameters.
        self.weight = self.bn.weight.detach().clone()
        self.bias = self.bn.bias.detach().clone()

        # Reset affine parameters to identify function.
        with torch.no_grad():
            self.bn.weight.fill_(1)
            self.bn.bias.fill_(0)
        self.bn.weight.requires_grad = False
        self.bn.bias.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Forward using writer code. The writer code is not passed as an argument,
        but is expected to be externally set under the `writer_code` attribute.

        Args:
            x (Tensor of shape (N, n_channels, h, w))
        """
        assert self.writer_code is not None, "Writer code not initialized."
        assert self.writer_code.ndim == 2 and self.writer_code.shape[0] == x.shape[0]

        bsz, n_channels = x.shape[:2]
        self.weight, self.bias = self.weight.to(x.device), self.bias.to(x.device)

        x = self.bn(x)

        weight_and_bias = self.adapt(self.writer_code)  # shape: (N, 2 * n_channels)
        weight_delta = weight_and_bias[:, :n_channels]
        bias_delta = weight_and_bias[:, n_channels:]

        weight = self.weight.unsqueeze(0).expand_as(weight_delta)
        bias = self.bias.unsqueeze(0).expand_as(bias_delta)
        weight = (weight + weight_delta).view(bsz, n_channels, 1, 1).expand_as(x)
        bias = (bias + bias_delta).view(bsz, n_channels, 1, 1).expand_as(x)

        return x * weight + bias

    @staticmethod
    def replace_bn_adaptive(
        module: nn.Module, writer_code_size: int, adaptation_num_hidden: int
    ):
        """
        Replace all nn.BatchNorm2d layers in a module with ConditionalBatchNorm2d layers.

        Returns:
            list of all newly added ConditionalBatchNorm2d modules
        """
        new_mods = []
        if isinstance(module, ConditionalBatchNorm2d):
            return new_mods
        if isinstance(module, nn.Sequential):
            for i, m in enumerate(module):
                if type(m) == nn.BatchNorm2d:
                    new_bn = ConditionalBatchNorm2d(
                        m, writer_code_size, adaptation_num_hidden
                    )
                    module[i] = new_bn
                    new_mods.append(new_bn)
        else:
            for attr_str in dir(module):
                attr = getattr(module, attr_str)
                if type(attr) == nn.BatchNorm2d:
                    new_bn = ConditionalBatchNorm2d(
                        attr, writer_code_size, adaptation_num_hidden
                    )
                    setattr(module, attr_str, new_bn)
                    new_mods.append(new_bn)

        for child_module in module.children():
            new_mods.extend(
                ConditionalBatchNorm2d.replace_bn_adaptive(
                    child_module, writer_code_size, adaptation_num_hidden
                )
            )
        return new_mods


class WriterAdaptiveResnet(nn.Module):
    """
    A Resnet where all batch normalization layers are replaced with writer-code
    adaptive layers. Concretely, this means the learned affine parameters of the
    batchnorm layers are replaced with parameters produced by a writer code
    transformation.
    """

    def __init__(
        self, resnet: nn.Module, writer_code_size: int, adaptation_num_hidden: int
    ):
        super().__init__()
        self.resnet = resnet
        # Replace batchnorm layers with adaptive ones.
        self.bn_layers = ConditionalBatchNorm2d.replace_bn_adaptive(
            self.resnet, writer_code_size, adaptation_num_hidden
        )

    def forward(self, imgs: Tensor, writer_code: torch.Tensor) -> torch.Tensor:
        # Set `writer_code` attribute for all ConditionalBatchNorm2d layers
        for l in self.bn_layers:
            l.writer_code = writer_code
        out = self.resnet(imgs)
        for l in self.bn_layers:
            l.writer_code = None
        return out


class WriterAdaptiveResnet2(nn.Module):
    """
    A Resnet where all batch normalization layers are replaced with
    adaptive batchnorm layers (CITE).
    """

    def __init__(
        self,
        resnet: nn.Module,
        layer_stats_per_writer: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    ):
        super().__init__()
        self.resnet = resnet
        self.layer_stats_per_writer = layer_stats_per_writer
        # Replace batchnorm layers with adaptive ones.
        self.bn_layers = AdaptiveBatchnorm2d.replace_bn_adaptive(
            self.resnet, self.layer_stats_per_writer
        )

    def forward(self, imgs: Tensor, writer_ids: torch.Tensor) -> torch.Tensor:
        # Set `writer_stats` attribute for all AdaptiveBatchnorm2d layers,
        # which contains the layer mean and variance for each writer in the current
        # batch.
        for i, bn_layer in enumerate(self.bn_layers):
            bn_layer.writer_ids = writer_ids
            # stats = []
            # for wid in writer_ids:
            #     wid_key = wid.item()
            #     mean = self.layer_stats_per_writer[wid_key][i]["mean"]
            #     std = self.layer_stats_per_writer[wid_key][i]["std"]
            #     stats.append([mean, std])
            # bn_layer.writer_stats = torch.tensor(stats, device=imgs.device)
        out = self.resnet(imgs)
        for l in self.bn_layers:
            l.writer_stats = None
        return out


class AdaptiveBatchnorm2d(nn.Module):
    def __init__(self, stats_per_writer: Tensor, weight: Tensor, bias: Tensor):
        super().__init__()
        assert stats_per_writer.ndim == 3
        self.register_buffer("stats_per_writer", stats_per_writer)  # (n_writers, C, 2)
        self.register_parameter("weight", weight)  # shape: (C,)
        self.register_parameter("bias", bias)  # shape: (C,)

        self.writer_ids = None  # to be set externally
        self.eps = 1e-5

    def forward(self, x: Tensor):
        assert x.ndim == 4, x.shape

        bsz, n_channels, h, w = x.shape

        # self.weight, self.bias = self.weight.to(x.device), self.bias.to(x.device)
        # self.stats_per_writer = self.stats_per_writer.to(x.device)

        stats = self.stats_per_writer[self.writer_ids]  # shape: (bsz, n_channels, 2)
        sub = stats[:, :, 0].view(bsz, n_channels, 1, 1).expand_as(x)
        div = torch.sqrt(
            stats[:, :, 1].view(bsz, n_channels, 1, 1).expand_as(x) + self.eps
        )
        x = (x - sub) / div

        weight = self.weight.view(1, n_channels, 1, 1).expand_as(x)
        bias = self.bias.view(1, n_channels, 1, 1).expand_as(x)

        return x * weight + bias

    @staticmethod
    def replace_bn_adaptive(module: nn.Module, stats_per_writer: Dict):
        """
        Replace all nn.BatchNorm2d layers in a module with AdaptiveBatchnorm2d layers.

        Returns:
            list of all newly added batchnorm modules
        """

        def dict2tensor(dct: Dict[int, Dict[int, Dict[str, float]]]):
            res = []
            for writer_id in dct.keys():
                wrtr_stats = []
                for chan, stats in dct[writer_id].items():
                    wrtr_stats.append([stats["mean"], stats["var"]])
                res.append(wrtr_stats)
            return torch.tensor(res, dtype=torch.float32)

        new_mods = []
        if isinstance(module, AdaptiveBatchnorm2d):
            return new_mods
        if isinstance(module, nn.Sequential):
            for i, m in enumerate(module):
                if type(m) == nn.BatchNorm2d:
                    stats_tns = dict2tensor(stats_per_writer[m.layer_idx])
                    new_bn = AdaptiveBatchnorm2d(stats_tns, m.weight, m.bias)
                    module[i] = new_bn
                    new_mods.append(new_bn)
        else:
            for attr_str in dir(module):
                attr = getattr(module, attr_str)
                if type(attr) == nn.BatchNorm2d:
                    stats_tns = dict2tensor(stats_per_writer[attr.layer_idx])
                    new_bn = AdaptiveBatchnorm2d(stats_tns, attr.weight, attr.bias)
                    setattr(module, attr_str, new_bn)
                    new_mods.append(new_bn)

        for child_module in module.children():
            new_mods.extend(
                AdaptiveBatchnorm2d.replace_bn_adaptive(child_module, stats_per_writer)
            )
        return new_mods


class AdaptiveBatchnormModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        layer_stats_per_writer: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
        d_model: int,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            layer_stats_per_writer: ...
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
        """
        super().__init__()
        self.layer_stats_per_writer = layer_stats_per_writer
        self.d_model = d_model

        if isinstance(base_model, FullPageHTREncoderDecoder):
            self.arch = "fphtr"
        elif isinstance(base_model, ShowAttendRead):
            self.arch = "sar"
        else:
            raise ValueError(f"Unrecognized model class: {base_model.__class__}")

        assert base_model.loss_fn.reduction == "mean"

        freeze(base_model)  # make sure the base model weights are frozen

        self.feature_transform = None
        resnet_old = (
            base_model.encoder if self.arch == "fphtr" else base_model.resnet_encoder
        )
        resnet_new = WriterAdaptiveResnet2(resnet_old, self.layer_stats_per_writer)
        if self.arch == "fphtr":
            base_model.encoder = resnet_new
        else:  # SAR
            base_model.resnet_encoder = resnet_new
        self.model = base_model

    def forward(
        self,
        imgs: Tensor,
        target: Tensor,
        writer_ids: Tensor,
        mode: TrainMode = TrainMode.TRAIN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        teacher_forcing = True if mode == TrainMode.TRAIN else False
        if self.arch == "fphtr":
            features = self.model.encoder(imgs, writer_ids)
            if teacher_forcing:
                logits = self.model.decoder.decode_teacher_forcing(features, target)
            else:
                logits, _ = self.model.decoder(features)
        else:  # SAR
            imgs = imgs.unsqueeze(1)
            features = self.model.resnet_encoder(imgs, writer_ids)
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
        sampled_ids = logits.argmax(-1)
        return logits, sampled_ids, loss
