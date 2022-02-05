import xml.etree.ElementTree as ET
import random
from pathlib import Path
from typing import Union, Any, List, Optional, Sequence, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import learn2learn as l2l
from torch.utils.data import Dataset
from torch import Tensor
from pytorch_lightning.callbacks import TQDMProgressBar


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


def filter_df_by_freq(df: pd.DataFrame, column: str, min_freq: int) -> pd.DataFrame:
    """
    Filters the DataFrame based on the value frequency in the specified column.

    Taken from https://stackoverflow.com/questions/30485151/python-pandas-exclude
    -rows-below-a-certain-frequency-count#answer-58809668.

    :param df: DataFrame to be filtered.
    :param column: Column name that should be frequency filtered.
    :param min_freq: Minimal value frequency for the row to be accepted.
    :return: Frequency filtered DataFrame.
    """
    # Frequencies of each value in the column.
    freq = df[column].value_counts()
    # Select frequent values. Value is in the index.
    frequent_values = freq[freq >= min_freq].index
    # Return only rows with value frequency above threshold.
    return df[df[column].isin(frequent_values)]


def identity_collate_fn(x: Sequence[Any]):
    """
    This function can be used for PyTorch dataloaders that return batches of size
    1 and do not require any collation of samples in the batch. This is useful if a
    batch of data is already prepared when it is passed to the dataloader.
    """
    assert len(x) == 1
    return x[0]


class PtTaskDataset(Dataset):
    def __init__(self, taskset: l2l.data.TaskDataset, epoch_length: int):
        super().__init__()
        self.taskset = taskset
        self.epoch_length = epoch_length

    def __getitem__(self, *args, **kwargs):
        return self.taskset.sample()

    def __len__(self):
        return self.epoch_length


def decode_prediction(
    pred: Tensor, label_encoder: LabelEncoder, eos_tkn_idx: int
) -> str:
    eos_idx = (pred == eos_tkn_idx).float().argmax().item()
    res = pred[:eos_idx] if eos_idx != 0 else pred
    return "".join(label_encoder.inverse_transform(res.tolist()))
