import xml.etree.ElementTree as ET
import random
import pickle
from pathlib import Path
from typing import Union, Any, List, Optional, Dict, Sequence

# from models import FullPageHTREncoderDecoder

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import learn2learn as l2l
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import TQDMProgressBar


# def gather_parameters(model: FullPageHTREncoderDecoder):
#     """Obtains a list of parameters that should be updated during MetaHTR training."""
#     parameters = []
#     modules_to_gather = [nn.Conv2d, ...]
#     modules = list(model.modules())
#     for m in modules
#         if any(isinstance(m, mod) for mod in modules_to_gather):
#             # TODO
#             ...
#     return parameters
#


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


def pickle_save(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)


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


class LabelEncoder:
    classes: Optional[List[str]]
    idx_to_cls: Optional[Dict[int, str]]
    cls_to_idx: Optional[Dict[str, int]]
    n_classes: Optional[int]

    def __init__(self):
        self.classes = None
        self.idx_to_cls = None
        self.cls_to_idx = None
        self.n_classes = None

    def transform(self, classes: Sequence[str]) -> List[int]:
        self.check_is_fitted()
        return [self.cls_to_idx[c] for c in classes]

    def inverse_transform(self, indices: Sequence[int]) -> List[str]:
        return [self.idx_to_cls[i] for i in indices]

    def fit(self, classes: Sequence[str]):
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.idx_to_cls = dict(enumerate(classes))
        self.cls_to_idx = {cls: i for i, cls in self.idx_to_cls.items()}
        return self

    def add_classes(self, classes: List[str]):
        new_classes = self.classes + classes
        assert len(set(new_classes)) == len(
            new_classes
        ), "New labels contain duplicates"
        return self.fit(new_classes)

    def read_encoding(self, filename: Union[str, Path]):
        if Path(filename).suffix == ".pkl":
            # Label encoding saved as Sklearn LabelEncoder instance.
            return self.read_sklearn_encoding(filename)
        else:
            classes = []
            saved_str = Path(filename).read_text()
            i = 0
            while i < len(saved_str):
                # This is a bit of a roundabout way to read the saved label encoding,
                # but it is necessary in order to read special characters (like `\n`)
                # correctly.
                c = saved_str[i]
                i += 1
                while i < len(saved_str) and saved_str[i] != "\n":
                    c += saved_str[i]
                    i += 1
                classes.append(c)
                i += 1
            return self.fit(classes)

    def read_sklearn_encoding(self, filename: Union[str, Path]):
        """
        Load an encoding from a Sklearn LabelEncoder pickle. This method exists to
        maintain backwards compatability with previously saved label encoders.
        """
        label_encoder = pickle_load(filename)
        classes = list(label_encoder.classes_)

        # Check if the to-be-saved encoding is correct.
        assert (
            list(label_encoder.inverse_transform(list(range(len(classes))))) == classes
        )
        self.fit(classes)
        self.dump(Path(filename).parent)
        return self

    def dump(self, outdir: Union[str, Path]):
        """Dump the encoded labels to a txt file."""
        out = "\n".join(cls for cls in self.classes)
        (Path(outdir) / "label_encoding.txt").write_text(out)

    def check_is_fitted(self):
        if self.idx_to_cls is None or self.cls_to_idx is None:
            raise ValueError("Label encoder is not fitted yet.")


class PtTaskDataset(Dataset):
    def __init__(self, taskset: l2l.data.TaskDataset, epoch_length: int):
        super().__init__()
        self.taskset = taskset
        self.epoch_length = epoch_length

    def __getitem__(self, *args, **kwargs):
        return self.taskset.sample()

    def __len__(self):
        return self.epoch_length


class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        for k in list(items.keys()):
            if k.startswith("grad"):
                items.pop(k, None)
        items.pop("v_num", None)
        return items
