from functools import partial
import math
from typing import Optional, Dict, Tuple, List, Callable, Any
from collections import defaultdict

from thesis.metahtr.util import LayerWiseLRTransform
from thesis.models import MAMLLearner
from thesis.util import (
    TrainMode,
    split_batch_for_adaptation,
    set_batchnorm_layers_train,
    set_dropout_layers_train,
    chunk_batch,
)

from htr.models.sar.sar import ShowAttendRead

import torch
import torch.nn as nn
import numpy as np
import learn2learn as l2l
from torch import Tensor
from torch.autograd import grad


class MAMLHTR(nn.Module, MAMLLearner):
    def __init__(
        self,
        base_model: nn.Module,
        transform: Optional[Callable] = None,
        ways: int = 8,
        shots: int = 16,
        inner_lr: float = 0.001,
        instance_weights_fn: Optional[Callable] = None,
        max_val_batch_size: int = 128,
        use_batch_stats_for_batchnorm: bool = False,
        use_dropout: bool = False,
        num_inner_steps: int = 1,
        allow_nograd: bool = False,
        **kwargs,
    ):
        """
        Implementation of vanilla MAML applied to Handwritten Text Recognition.

        Some implementation details to note:
        - for batchnorm layers, stored statistics are used by default during
          both inner and outer loop, but this can be changed to batch statistics using
          the `--use_batch_stats_for_batchnorm` argument (specified in the PL module).

        Args:
            base_model (nn.Module): base model to apply MAML on
            transform (Optional[Callable]): transform used to update the module. See
                learn2learn docs for more details.
            ways (int): number of writers per batch
            shots (int): number of samples per writer per batch
            inner_lr (float): learning rate used in the inner loop
            instance_weights_fn (Callable): function to calculate instance-specific
                weights.
            use_batch_stats_for_batchnorm (bool): if True, batch statistics will be
                used for batchnorm layers, rather than stored statistics
            use_dropout (bool): whether to use dropout in the outer loop.
            num_inner_steps (int): how many inner loop optimization steps to perform
            allow_nograd (bool): Whether to allow adaptation with parameters that
                have `requires_grad = False`.
        """
        super().__init__()

        assert num_inner_steps >= 1

        self.ways = ways
        self.shots = shots
        self.inner_lr = inner_lr
        self.instance_weights_fn = instance_weights_fn
        self.max_val_batch_size = max_val_batch_size
        self.use_batch_stats_for_batchnorm = use_batch_stats_for_batchnorm
        self.use_dropout = use_dropout
        self.num_inner_steps = num_inner_steps
        self.allow_nograd = allow_nograd

        if transform is not None:
            self.gbml = l2l.algorithms.GBML(
                base_model,
                transform=transform,
                first_order=False,
                allow_unused=True,
                allow_nograd=allow_nograd,
            )
        else:
            self.gbml = l2l.algorithms.MAML(
                base_model,
                lr=inner_lr,
                first_order=False,
                allow_unused=True,
                allow_nograd=allow_nograd,
            )

        self.ignore_index = self.gbml.module.pad_tkn_idx
        # Disabling CuDNN is necessary for RNNs due to limitations in the CuDNN API
        # for double backward passes.
        self.use_cudnn = not isinstance(base_model, ShowAttendRead)
        self.char_to_avg_inst_weight = None

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

        # For some reason using an autograd context manager like torch.enable_grad()
        # here does not work, perhaps due to some unexpected interaction between
        # Pytorch Lightning and the learn2learn lib. Therefore gradient
        # calculation should be set beforehand, outside of the current function.

        # Adapt the model.
        learner, _, _ = self.fast_adaptation(
            learner, adaptation_imgs, adaptation_targets
        )

        # Run inference on the adapted model.
        with torch.inference_mode():
            logits, sampled_ids, _ = learner(inference_imgs)

        return logits, sampled_ids

    def meta_learn(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.TRAIN
    ) -> Tuple[Tensor, float, Optional[Dict[int, List]]]:
        outer_loss, n_query_images = 0.0, 0
        inner_losses = []
        char_to_inst_weights = defaultdict(list)

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
                learner, support_loss, instance_weights = self.fast_adaptation(
                    learner, support_imgs, support_tgts
                )

                inner_losses.append(support_loss.item())
                if instance_weights is not None:
                    # Store the instance-specific weights for logging.
                    ignore_mask = support_tgts == self.ignore_index
                    for tgt, w in zip(support_tgts[~ignore_mask], instance_weights):
                        char_to_inst_weights[tgt.item()].append(w.item())

            # Outer loop.
            loss_fn = learner.module.loss_fn
            reduction = loss_fn.reduction
            loss_fn.reduction = "mean"
            if mode is TrainMode.TRAIN:
                set_dropout_layers_train(learner, self.use_dropout)
                with torch.backends.cudnn.flags(enabled=self.use_cudnn):
                    _, query_loss = learner.module.forward_teacher_forcing(
                        query_imgs, query_tgts
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
                        _, preds, query_loss = learner(img, tgt)

                    # Calculate metrics.
                    self.gbml.module.cer_metric(preds, tgt)
                    self.gbml.module.wer_metric(preds, tgt)
                    outer_loss += query_loss * img.size(0)
            loss_fn.reduction = reduction

        outer_loss /= n_query_images
        inner_loss_avg = np.mean(inner_losses)

        return outer_loss, inner_loss_avg, char_to_inst_weights

    def fast_adaptation(
        self,
        learner: l2l.algorithms.GBML,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
    ) -> Tuple[Any, float, Optional[Tensor]]:
        """
        Takes a single gradient step on a batch of data.
        """
        learner.train()
        set_dropout_layers_train(learner, False)  # disable dropout
        set_batchnorm_layers_train(learner, self.use_batch_stats_for_batchnorm)

        with torch.backends.cudnn.flags(enabled=self.use_cudnn):
            _, support_loss_unreduced = learner.module.forward_teacher_forcing(
                adaptation_imgs, adaptation_targets
            )

        ignore_mask = adaptation_targets == self.ignore_index
        instance_weights = None
        if self.instance_weights_fn is not None:
            instance_weights_fn = self.instance_weights_fn
            instance_weights = instance_weights_fn(
                learner=learner,
                loss_unreduced=support_loss_unreduced,
                ignore_mask=ignore_mask,
            )
            support_loss = torch.sum(
                support_loss_unreduced[~ignore_mask] * instance_weights
            ) / adaptation_imgs.size(0)
        else:
            support_loss = torch.mean(support_loss_unreduced[~ignore_mask])

        # Calculate gradients and take an optimization step.
        learner.adapt(support_loss)

        return learner, support_loss, instance_weights


class MetaHTR(MAMLHTR):
    def __init__(
        self,
        base_model: nn.Module,
        num_clf_weights: int,
        inst_mlp_hidden_size: int = 8,
        initial_inner_lr: float = 0.001,
        use_instance_weights: bool = True,
        **kwargs,
    ):
        self.num_clf_weights = num_clf_weights
        self.inst_mlp_hidden_size = inst_mlp_hidden_size
        self.initial_inner_lr = initial_inner_lr
        self.use_instance_weights = use_instance_weights

        instance_weights_fn = None
        inst_w_mlp = None
        if use_instance_weights:
            inst_w_mlp = nn.Sequential(  # instance-specific weight MLP
                nn.Linear(
                    num_clf_weights * 2,
                    inst_mlp_hidden_size,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(inst_mlp_hidden_size, inst_mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(inst_mlp_hidden_size, 1),
                nn.Sigmoid(),
            )
            instance_weights_fn = partial(
                self.calculate_instance_specific_weights, inst_w_mlp=inst_w_mlp
            )
        transform = LayerWiseLRTransform(initial_inner_lr)

        super().__init__(
            base_model=base_model,
            transform=transform,
            instance_weights_fn=instance_weights_fn,
            **kwargs,
        )
        self.inst_w_mlp = inst_w_mlp

    @staticmethod
    def calculate_instance_specific_weights(
        inst_w_mlp: nn.Module,
        learner: l2l.algorithms.GBML,
        loss_unreduced: Tensor,
        ignore_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculates instance-specific weights, based on the per-instance gradient
        w.r.t to the final classifcation layer.

        Args:
            inst_w_mlp (nn.Module): NN to calculate the instance weights
            learner (l2l.algorithms.GBML): learn2learn GBML learner
            loss_unreduced (Tensor): tensor of shape (B*T,), where B = batch size and
                T = maximum sequence length in the batch, containing the per-instance
                loss, i.e. the loss for each decoding time step.
            ignore_mask (Optional[Tensor]): mask of the same shape as
                `loss_unreduced`, specifying what values to ignore for the loss

        Returns:
            Tensor of shape (B*T,), containing the instance specific weights
        """
        grad_inputs = []

        if ignore_mask is not None:
            assert (
                ignore_mask.shape == loss_unreduced.shape
            ), "Mask should have the same shape as the loss tensor."
        else:
            ignore_mask = torch.zeros_like(loss_unreduced)
        ignore_mask = ignore_mask.bool()
        mean_loss = loss_unreduced[~ignore_mask].mean()

        mean_loss_grad = grad(
            mean_loss,
            learner.module.clf_layer.weight,
            create_graph=True,
            retain_graph=True,
        )[0]
        # It is not ideal to have to compute gradients like this in a loop - which
        # loses the benefit of parallelization -, but unfortunately Pytorch does not
        # provide any native functonality for calculating per-example gradients.
        for instance_loss in loss_unreduced[~ignore_mask]:
            instance_grad = grad(
                instance_loss,
                learner.module.clf_layer.weight,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_inputs.append(
                torch.cat([instance_grad.flatten(), mean_loss_grad.flatten()])
            )
        grad_inputs = torch.stack(grad_inputs, 0)
        instance_weights = inst_w_mlp(grad_inputs)
        assert instance_weights.numel() == torch.sum(~ignore_mask)

        return instance_weights
