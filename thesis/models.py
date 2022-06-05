from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import learn2learn as l2l
from torch import Tensor

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead

from thesis.util import freeze, TrainMode, chunk_batch, set_batchnorm_layers_train


class MAMLLearner(ABC):
    @abstractmethod
    def meta_learn(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.TRAIN
    ) -> Tuple[Tensor, Tensor, Optional[Dict[int, List]]]:
        """Process a single batch of tasks."""
        pass

    @abstractmethod
    def fast_adaptation(
        self,
        learner: l2l.algorithms.GBML,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
    ) -> Tuple[Any, float, Optional[Tensor]]:
        """Take a single gradient step on a batch of data."""
        pass

    @abstractmethod
    def forward(
        self,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
        inference_imgs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pass


class FewShotFinetuningModel(nn.Module):
    """
    Use a batch of adaptation data to perform finetuning on a subset of the model
    parameters (e.g. the final classification layer).
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        learning_rate_finetune: float,
        shots: int = 16,
        max_val_batch_size: int = 64,
        finetune_opt_steps: int = 1,
        use_adam_for_adaptation: bool = False,
    ):
        """
        Args:
            base_model (nn.Module): pre-trained HTR model, frozen during adaptation
            d_model (int): size of the feature vectors produced by the feature
                extractor (e.g. CNN).
            learning_rate_finetune (float): learning rate used for fast adaptation of an
                initial embedding during val/test
            shots (int): number of samples to use for finetuning
            max_val_batch_size (int): maximum val batch size
            finetune_opt_steps (int): number of optimization steps during finetuning
            use_adam_for_adaptation (bool): whether to use Adam during adaptation
                (otherwise plain SGD is used)
        """
        assert base_model.loss_fn.reduction == "mean"

        super().__init__()
        self.base_model = base_model
        self.d_model = d_model
        self.learning_rate_finetune = learning_rate_finetune
        self.shots = shots
        self.max_val_batch_size = max_val_batch_size
        self.finetune_opt_steps = finetune_opt_steps
        self.use_adam_for_adaptation = use_adam_for_adaptation

        if isinstance(self.base_model, FullPageHTREncoderDecoder):
            self.arch = "fphtr"
        elif isinstance(self.base_model, ShowAttendRead):
            self.arch = "sar"
        else:
            raise ValueError(f"Unrecognized model class: {self.base_model.__class__}")

        # Save the original weights of the final classification layer.
        self.old_clf_weight = self.base_model.decoder.clf.weight.detach().clone()
        self.old_clf_bias = self.base_model.decoder.clf.bias.detach().clone()
        self.base_model.decoder.clf.requires_grad_(True)  # finetune the last layer
        freeze(self.base_model)  # make sure the base model weights are frozen

    def forward(
        self,
        adapt_imgs: Tensor,
        adapt_tgts: Tensor,
        inference_imgs: Tensor,
        inference_tgts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Finetune on an adaptation batch, and run inference on another batch.

        Args:
            adapt_imgs (Tensor): images to do adaptation on
            adapt_tgts (Tensor): targets for `adaptation_imgs`
            inference_imgs (Tensor): images to make predictions on
            inference_tgts (Optional[Tensor]): targets for `inference_imgs`
        """
        # Finetune the model on the adaptation data.
        self.finetune(adapt_imgs, adapt_tgts)

        # The set of writer examples may be too large too fit into a single
        # batch. Therefore, chunk the data and process each chunk individually.
        # TODO: this does not work if inference_tgts=None
        inf_img_chunks, inf_tgt_chunks = chunk_batch(
            inference_imgs, inference_tgts, self.max_val_batch_size
        )

        # Run inference on the finetuned model.
        inference_loss = 0.0
        all_logits = []
        for img, tgt in zip(inf_img_chunks, inf_tgt_chunks):
            with torch.inference_mode():
                logits, _, loss = self.base_model(img, tgt)
            all_logits.append(logits)
            inference_loss += loss * img.size(0)
        inference_loss /= inference_imgs.size(0)
        max_seq_len = max(t.size(1) for t in all_logits)
        logits = torch.cat(
            [F.pad(t, (0, 0, 0, max_seq_len - t.size(1))) for t in all_logits], 0
        )
        # TODO: is it okay to pad logits with zeros?
        sampled_ids = logits.argmax(-1)
        return logits, sampled_ids, inference_loss

    def finetune(self, adaptation_imgs: Tensor, adaptation_targets: Tensor):
        """Finetune the base model on adaptation data."""
        model = self.base_model
        # Reset the previously finetuned base model parameters.
        self.reset_params(adaptation_imgs.device)
        # Freeze batchnorm stats and use stored statistics.
        set_batchnorm_layers_train(model, False)
        # Set up optimizer.
        if self.use_adam_for_adaptation:
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_finetune)
        else:
            # Plain SGD, i.e. update rule `g = g - lr * grad(loss)`
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate_finetune)

        # Finetune the model.
        print("Finetuning writer.")
        for i in range(self.finetune_opt_steps):
            logits, loss = model.forward_teacher_forcing(
                adaptation_imgs, adaptation_targets
            )
            print(f"Train loss at step {i}: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def reset_params(self, device):
        """Reset the model parameters to their initial values before finetuning.
        Note that this does not mean resetting the weights to random values,
        but rather undoing the finetuning by using the originally stored parameters."""
        with torch.no_grad():
            self.base_model.decoder.clf.weight.data = self.old_clf_weight.to(device)
            self.base_model.decoder.clf.bias.data = self.old_clf_bias.to(device)
        self.base_model.decoder.clf.requires_grad_(True)
