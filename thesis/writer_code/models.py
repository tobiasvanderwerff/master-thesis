from enum import Enum
from functools import partial
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead

from thesis.util import freeze


class WriterEmbeddingType(Enum):
    """
    Type of writer embedding. Choices:

    learned: an embedding whose weights are learned with backpropagation
    transformed: an embedding produced by some transform of features, e.g. a feature
        map passed through a dense layer to produce the embedding.
    """

    LEARNED = 1
    TRANSFORMED = 2

    @staticmethod
    def from_string(s: str):
        s = s.lower()
        if s == "learn":
            return WriterEmbeddingType.LEARNED
        elif s == "transform":
            return WriterEmbeddingType.TRANSFORMED
        else:
            raise ValueError(f"{s} is not a valid embedding method.")


class WriterCodeAdaptiveModel(nn.Module):
    """
    Implementation of speaker-adaptive model by Abdel-Hamid et al. (2013),
    adapted to HTR models.
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        emb_size: int,
        num_hidden: int,
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
            num_hidden (int): hidden size for adaptation MLP
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
        self.num_hidden = num_hidden
        self.num_writers = num_writers
        self.learning_rate_emb = learning_rate_emb
        self.embedding_type = embedding_type
        self.adaptation_opt_steps = adaptation_opt_steps
        self.use_adam_for_adaptation = use_adam_for_adaptation

        self.writer_embs = nn.Embedding(num_writers, emb_size)
        self.adaptation = AdaptationMLP(d_model, emb_size, num_hidden)
        self.base_model_with_adaptation = BaseModelAdaptation(base_model)

        assert all(p.requires_grad for p in self.writer_embs.parameters())
        assert all(p.requires_grad for p in self.adaptation.parameters())
        assert base_model.loss_fn.reduction == "mean"

        freeze(base_model)  # make sure the base model weights are frozen
        # Finetune the linear layer in the base model directly following the adaptation
        # model.
        base_model.encoder.linear.requires_grad_(True)

    def forward(
        self, *args, mode: str = "train", **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert mode in ["train", "val", "test"]
        if mode == "train":  # use a pre-trained embedding
            logits, loss = self.forward_existing_code(*args, **kwargs)
        else:  # initialize and train a new writer embedding
            logits, loss = self.forward_new_code(*args, **kwargs)
        sampled_ids = logits.argmax(-1)
        return logits, sampled_ids, loss

    def forward_existing_code(
        self, imgs: Tensor, target: Tensor, writer_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Load an existing writer code.
        writer_emb = self.writer_embs(writer_ids)  # (N, emb_size)

        # Run inference using the writer code.
        intermediate_transform = partial(self.adaptation.forward, writer_emb=writer_emb)
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
        Adapt on a set of images for a particular writer and run inference on another set.

        Args:
            adapt_imgs (Tensor): images to do adaptation on
            adapt_tgts (Tensor): targets for `adaptation_imgs`
            inference_imgs (Tensor): images to make predictions on
            inference_tgts (Optional[Tensor]): targets for `inference_imgs`
        Returns:
            predictions on `inference_imgs`, in the form of a 3-tuple:
                - logits, obtained at each time step during decoding
                - mean loss
        """
        # Train a writer code.
        writer_code = self.new_writer_code(adapt_imgs, adapt_tgts)

        # Run inference using the writer code.
        with torch.inference_mode():
            intermediate_transform = partial(
                self.adaptation.forward, writer_emb=writer_code
            )
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
        writer_emb = torch.empty(1, self.emb_size).to(adaptation_imgs.device)
        writer_emb.normal_()  # mean 0, std 1
        writer_emb.requires_grad = True

        if self.use_adam_for_adaptation:
            optimizer = optim.Adam(iter([writer_emb]), lr=self.learning_rate_emb)
        else:
            # Plain SGD, i.e. update rule `g = g - lr * grad(loss)`
            optimizer = optim.SGD(iter([writer_emb]), lr=self.learning_rate_emb)

        for _ in range(self.adaptation_opt_steps):
            intermediate_transform = partial(
                self.adaptation.forward, writer_emb=writer_emb
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


class BaseModelAdaptation(nn.Module):
    """
    Base HTR model with an adaptation model injected in between.

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
