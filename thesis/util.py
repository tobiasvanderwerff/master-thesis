from copy import copy
from enum import Enum
from functools import partial
import math
from pathlib import Path
import shutil
from typing import Optional, Tuple, List, Sequence, Any, Union, Dict
from collections import OrderedDict, defaultdict

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import load_hparams_from_yaml

from htr.data import IAMDataset
from htr.util import LabelEncoder

import torch
import learn2learn as l2l
import torch.nn as nn
import numpy as np
import pandas as pd
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

PREDICTIONS_TO_LOG = {
    "word": 10,
    "line": 6,
    "form": 1,
}
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"


class ExtendedEnum(Enum):
    @classmethod
    def from_string(cls, s: str):
        s = s.lower()
        for el in cls:  # iterate over all enum values
            if s == el.name.lower():
                return el
        raise ValueError(f"{s} is not a valid enum specifier.")


class TrainMode(ExtendedEnum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class BaseModelArch(ExtendedEnum):
    FPHTR = 1
    SAR = 2


def main_lit_models():
    from thesis.lit_models import LitMAMLLearner
    from thesis.metahtr.lit_models import LitMetaHTR
    from thesis.writer_code.lit_models import (
        LitWriterCodeAdaptiveModel,
        LitWriterCodeAdaptiveModelMAML,
    )

    return {
        "MAML": LitMAMLLearner,
        "MetaHTR": LitMetaHTR,
        "WriterCodeAdaptiveModel": LitWriterCodeAdaptiveModel,
        "WriterCodeAdaptiveModelMAML": LitWriterCodeAdaptiveModelMAML,
    }


class MainModelArch(ExtendedEnum):
    MAML = 1
    MetaHTR = 2
    WriterCodeAdaptiveModel = 3
    WriterCodeAdaptiveModelMAML = 4


def get_label_encoder(trained_model_path: Union[str, Path]) -> LabelEncoder:
    """Load a stored label encoder originating from a trained model."""
    assert Path(
        trained_model_path
    ).is_file(), f"{trained_model_path} does not point to a file."
    model_path = Path(trained_model_path).resolve()
    le_path_1 = model_path.parent.parent / "label_encoding.txt"
    le_path_2 = model_path.parent.parent / "label_encoder.pkl"
    assert le_path_1.is_file() or le_path_2.is_file(), (
        f"Label encoder file not found at {le_path_1} or {le_path_2}. "
        f"Make sure 'label_encoding.txt' exists in the lightning_logs directory."
    )
    le_path = le_path_2 if le_path_2.is_file() else le_path_1
    return LabelEncoder().read_encoding(le_path)


def save_label_encoder(label_encoder: LabelEncoder, path: Union[str, Path]):
    """Save a label encoder to a specified path."""
    Path(path).mkdir(exist_ok=True, parents=True)
    label_encoder.dump(path)


def get_pl_tb_logger(log_dir: Union[str, Path], version: Optional[str] = None):
    """Get a Pytorch Lightning Tensorboard logger."""
    return pl_loggers.TensorBoardLogger(str(Path(log_dir)), name="", version=version)


def copy_hyperparameters_to_logging_dir(
    trained_model_path: Union[str, Path], log_dir: Union[str, Path]
) -> Tuple[str, Dict[str, Any]]:
    """
    Copy hyper-parameters to logging directory. The loaded base model has an associated
    `hparams.yaml` file, which is copied to the current logging directory so that the base model can be
    loaded later using the saved hyper parameters.

    Returns:
        - path to hparams file
        - loaded hyperparameters as a dictionary
    """
    model_path = Path(trained_model_path)
    model_path_1 = model_path.parent.parent / "model_hparams.yaml"
    model_path_2 = model_path.parent.parent / "hparams.yaml"
    model_hparams_file = model_path_1 if model_path_1.is_file() else model_path_2
    model_hparams_file = str(model_hparams_file)
    shutil.copy(model_hparams_file, Path(log_dir) / "model_hparams.yaml")
    return model_hparams_file, load_hparams_from_yaml(model_hparams_file)


def prepare_iam_splits(dataset: IAMDataset, aachen_splits_path: Union[str, Path]):
    """Prepare IAM dataset train/val/(test) splits.

    The Aachen splits are used for the IAM dataset. It should be noted that these
    splits do not encompass the complete IAM dataset. Also worth noting is that in the
    Aachen splits, the writers present in train/val/test are disjoint.
    """
    train_splits = (aachen_splits_path / "train.uttlist").read_text().splitlines()
    validation_splits = (
        (aachen_splits_path / "validation.uttlist").read_text().splitlines()
    )
    test_splits = (aachen_splits_path / "test.uttlist").read_text().splitlines()

    data_train = dataset.data[dataset.data["img_id"].isin(train_splits)]
    data_val = dataset.data[dataset.data["img_id"].isin(validation_splits)]
    data_test = dataset.data[dataset.data["img_id"].isin(test_splits)]

    ds_train = copy(dataset)
    ds_train.data = data_train

    ds_val = copy(dataset)
    ds_val.data = data_val

    ds_test = copy(dataset)
    ds_test.data = data_test

    return ds_train, ds_val, ds_test


def prepare_l2l_taskset(
    dataset: IAMDataset,
    ways: int,
    bookkeeping_path: Union[str, Path],
    shots: Optional[int] = None,
    is_train: bool = True,
):
    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = dataset.label_enc.transform(
        [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]
    )
    collate_fn = partial(
        IAMDataset.collate_fn,
        pad_val=pad_tkn_idx,
        eos_tkn_idx=eos_tkn_idx,
        dataset_returns_writer_id=True,
    )

    # Setting the _bookkeeping_path attribute will make the MetaDataset instance
    # load its label-index mapping from a file, rather than creating it (which takes a
    # long time). If the path does not exists, the bookkeeping will be created and
    # stored on disk afterwards. Number of shots is stored along with the mapping,
    # because due to filtering of writers with less than `shots * 2` examples,
    # the mapping can change with the number of shots.
    dataset._bookkeeping_path = bookkeeping_path
    dataset_meta = l2l.data.MetaDataset(dataset)

    # Define learn2learn task transforms.
    task_trnsf = [
        # Nways picks N random labels (writers in this case)
        l2l.data.transforms.NWays(dataset_meta, n=ways),
        # Load the data.
        l2l.data.transforms.LoadData(dataset_meta),
    ]
    if shots is not None:
        # Keep K samples for each present writer.
        task_trnsf.insert(1, l2l.data.transforms.KShots(dataset_meta, k=shots))

    taskset = l2l.data.TaskDataset(
        dataset_meta,
        task_trnsf,
        num_tasks=-1,
        task_collate=collate_fn,
    )

    #
    sample_per_epoch = int(len(dataset.writer_ids) / ways)

    # Wrap the task datasets into a simple class that sets a length for the dataset
    # (other than 1, which is the default if setting num_tasks=-1).
    # This is necessary because the dataset length is used by Pytorch dataloaders to
    # determine how many batches are in the dataset per epoch.
    return PtTaskDataset(taskset, epoch_length=sample_per_epoch)


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


def get_parameter_names(checkpoint_path: Union[str, Path]):
    ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return [wn for wn in ckpt["state_dict"].keys()]


def chunk_batch(
    imgs: Tensor, targets: Tensor, max_batch_size: int
) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
    n_chunks = math.ceil(imgs.size(0) / max_batch_size)
    query_img_chunks = torch.chunk(imgs, n_chunks)
    query_tgt_chunks = torch.chunk(targets, n_chunks)
    return query_img_chunks, query_tgt_chunks


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
        imgs_, target_ = (
            imgs[task_slice],
            target[task_slice],
        )
        if limit_num_samples_per_task is not None:
            imgs_, target_ = (
                imgs[:limit_num_samples_per_task],
                target[:limit_num_samples_per_task],
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


def set_batchnorm_layers_train(model: nn.Module, training: bool = True):
    _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d)
    for m in model.modules():
        if isinstance(m, _batchnorm_layers):
            m.training = training


@torch.no_grad()
def batchnorm_reset_running_stats(model: nn.Module):
    _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d)
    for m in model.modules():
        if isinstance(m, _batchnorm_layers):
            m.reset_running_stats()


def set_dropout_layers_train(model: nn.Module, training: bool = True):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = training


def reset_all_weights(module: nn.Module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif hasattr(module, "_reset_parameters"):
        module._reset_parameters()
    for m in module.children():
        reset_all_weights(m)


def load_best_pl_checkpoint(
    trainer: "pl.Trainer",
    pl_module: "pl.LightningModule",
    label_encoder: LabelEncoder,
    verbose: bool = True,
):
    """
    Load the best checkpoint for a given PL module, and set it as new model for
    the given trainer.
    """
    ckpt_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_callback = cb
            break
    assert ckpt_callback is not None, "ModelCheckpoint not found in callbacks."
    best_model_path = ckpt_callback.best_model_path

    if verbose:
        print(f"Loading best model at {best_model_path}")

    model_hparams_file = Path(best_model_path).parent.parent / "model_hparams.yaml"
    args = dict(
        base_model_arch=pl_module.base_model_arch,
        main_model_arch=pl_module.main_model_arch,
        checkpoint_path=best_model_path,
        model_hparams_file=model_hparams_file,
        label_encoder=label_encoder,
        writer_codes=pl_module.writer_codes,
        num_workers=pl_module.num_workers,
    )
    args.update(pl_module.hparams)

    cls = pl_module.__class__
    model = cls.init_with_base_model_from_checkpoint(**args)
    trainer.model = model


def get_bn_statistics(model: nn.Module, dataloader: DataLoader, device: str = "cuda:0"):
    # Collect statistics.
    model = model.to(device)
    tracker = BNStatisticTracker(model)
    # CNT = 0
    bn_layers = collect_bn_layers(model)
    for data in dataloader:
        imgs, target, writer_ids = data
        imgs, target, writer_ids = (
            imgs.to(device),
            target.to(device),
            writer_ids.to(device),
        )

        # Set the writer ids as a layer attribute (because it cannot be passed as an
        # argument).
        for i, bn_layer in enumerate(bn_layers):
            bn_layer.writer_ids = writer_ids

        with torch.inference_mode():
            _ = model.forward_teacher_forcing(imgs, target)
        # CNT += 1
        # if CNT == 20:
        #     break
        # TODO: remove
    tracker.remove_hooks()

    # Aggregate statistics.
    layer_stats = tracker.activ_stats_per_writer
    for lname in layer_stats.keys():
        for wid in layer_stats[lname].keys():
            for chan, lstats in layer_stats[lname][wid].items():
                mean = np.mean(lstats["mean"])
                var = np.mean(lstats["var"])
                layer_stats[lname][wid][chan]["mean"] = mean
                layer_stats[lname][wid][chan]["var"] = var
    return layer_stats


class BNStatisticTracker(nn.Module):
    """
    Collects activation statistics for each batchnorm layer in a model,
    using forward hooks.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activ_stats_per_writer = defaultdict(dict)
        self.fhooks = []

        cnt = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.fhooks.append(m.register_forward_hook(self.forward_hook_bn(cnt)))
                cnt += 1

    def forward_hook_bn(self, layer_idx: int):
        def hook(module: nn.BatchNorm2d, input: Any, output: Any):
            inpt = input[0]
            n_channels = inpt.shape[1]
            writer_ids = module.writer_ids
            # set the layer index as an attribute so that it can be referenced later.
            module.layer_idx = layer_idx
            for i, wid in enumerate(writer_ids):
                ft_maps = inpt[i]
                wid_key = wid.item()
                if self.activ_stats_per_writer[layer_idx].get(wid_key) is None:
                    self.activ_stats_per_writer[layer_idx][wid_key] = {
                        chan: {"mean": [], "var": []} for chan in range(n_channels)
                    }
                # TODO: weight the mean and var by the amount of units, rather than
                #  weighing each feature map equally.
                for chan in range(n_channels):
                    ft_map = ft_maps[chan]
                    dct = self.activ_stats_per_writer[layer_idx][wid_key][chan]
                    dct["mean"].append(ft_map.mean().item())
                    dct["var"].append(ft_map.var().item())

        return hook

    def remove_hooks(self):
        for hook in self.fhooks:
            hook.remove()


def collect_bn_layers(module: nn.Module):
    """
    Collect all instances of nn.BatchNorm2d in a module.
    """
    res = []
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            res.append(m)
    # if isinstance(module, nn.BatchNorm2d):
    #     return res
    # if isinstance(module, nn.Sequential):
    #     for i, m in enumerate(module):
    #         if type(m) == nn.BatchNorm2d:
    #             res.append(m)
    # else:
    #     for attr_str in dir(module):
    #         attr = getattr(module, attr_str)
    #         if type(attr) == nn.BatchNorm2d:
    #             res.append(attr)
    #
    # for child_module in module.children():
    #     res.extend(collect_bn_layers(child_module))
    return res
