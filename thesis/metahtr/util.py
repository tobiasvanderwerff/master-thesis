from typing import Any, Sequence

import torch.nn as nn
import learn2learn as l2l


class LayerWiseLRTransform:
    """
    A modified version of the l2l.optim.ModuleTransform class, meant to facilitate
    per-layer learning rates in the MAML framework.
    """

    def __init__(self, initial_lr: float = 0.0001):
        self.initial_lr = initial_lr

    def __call__(self, parameter):
        # in combination with `GBML` class, `l2l.nn.Scale` will scale the gradient for
        # each layer in a model with an adaptable learning rate.
        transform = l2l.nn.Scale(shape=1, alpha=self.initial_lr)
        numel = parameter.numel()
        flat_shape = (1, numel)
        return l2l.optim.ReshapedTransform(
            transform=transform,
            shape=flat_shape,
        )


def set_norm_layers_to_train(module: nn.Module):
    """
    Use batch statistics rather than running statistics for normalization
    layers (batchnorm, layernorm).
    """
    for n, m in module.named_modules():
        mn = n.split(".")[-1]
        if "bn" in mn or "norm" in mn:
            m.training = True
