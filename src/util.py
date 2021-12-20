import xml.etree.ElementTree as ET
import random
import pickle
from pathlib import Path
from typing import Union, Sequence, Any

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import TQDMProgressBar


def pickle_save(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)


class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def read_xml(xml_file: Union[Path, str]) -> ET.Element:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root


def find_child_by_tag(
    xml_el: ET.Element, tag: str, value: str
) -> Union[ET.Element, None]:
    for child in xml_el:
        if child.get(tag) == value:
            return child
    return None


def matplotlib_imshow(img: torch.Tensor, one_channel=True):
    assert img.device.type == "cpu"
    if one_channel and img.ndim == 3:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
