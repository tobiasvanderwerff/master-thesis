from typing import Optional, Tuple, List, Sequence, Any

from htr.util import LabelEncoder

import torch
import learn2learn as l2l
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor


class PtTaskDataset(Dataset):
    def __init__(self, taskset: l2l.data.TaskDataset, epoch_length: int):
        super().__init__()
        self.taskset = taskset
        self.epoch_length = epoch_length

    def __getitem__(self, *args, **kwargs):
        return self.taskset.sample()

    def __len__(self):
        return self.epoch_length


def identity_collate_fn(x: Sequence[Any]):
    """
    This function can be used for PyTorch dataloaders that return batches of size
    1 and do not require any collation of samples in the batch. This is useful if a
    batch of data is already prepared when it is passed to the dataloader.
    """
    assert len(x) == 1
    return x[0]


def decode_prediction(
    pred: Tensor, label_encoder: LabelEncoder, eos_tkn_idx: int
) -> str:
    eos_idx = (pred == eos_tkn_idx).float().argmax().item()
    res = pred[:eos_idx] if eos_idx != 0 else pred
    return "".join(label_encoder.inverse_transform(res.tolist()))


def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def split_batch_for_adaptation(
    batch, ways: int, shots: int, limit_num_samples_per_task: Optional[int] = None
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Split a batch of data for adaptation.

    Based on a batch of the form (imgs, target, writer_ids), split the batch based on
    the writers in the batch, and then split each writer batch into a adaptation and
    query batch.

    Returns:
        Adaptation/query 4-tuples for each writer, of the form
            (adaptation_imgs, adaptation_tgts, query_imgs, query_tgts)
    """
    imgs, target, writer_ids = batch
    writer_ids_uniq = writer_ids.unique().tolist()

    assert len(writer_ids_uniq) == ways, f"{len(writer_ids_uniq)} vs {ways}"

    # Split the batch into N different writers, where N = ways.
    writer_batches = []
    for task in range(ways):  # tasks correspond to different writers
        wrtr_id = writer_ids_uniq[task]
        task_slice = writer_ids == wrtr_id
        imgs_, target_, writer_ids_ = (
            imgs[task_slice],
            target[task_slice],
            writer_ids[task_slice],
        )
        if limit_num_samples_per_task is not None:
            imgs_, target_, writer_ids_ = (
                imgs[:limit_num_samples_per_task],
                target[:limit_num_samples_per_task],
                writer_ids[:limit_num_samples_per_task],
            )

        # Separate data into support/query set.
        adaptation_indices = np.zeros(imgs_.size(0), dtype=bool)
        # Select first k even indices for adaptation set.
        adaptation_indices[np.arange(shots) * 2] = True
        # Select remaining indices for query set.
        query_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_imgs, adaptation_tgts = (
            imgs_[adaptation_indices],
            target_[adaptation_indices],
        )
        query_imgs, query_tgts = imgs_[query_indices], target_[query_indices]
        writer_batches.append(
            (adaptation_imgs, adaptation_tgts, query_imgs, query_tgts)
        )

    return writer_batches


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
