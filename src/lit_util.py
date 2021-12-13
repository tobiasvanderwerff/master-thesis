from typing import Optional, Dict, Any
from collections import OrderedDict

import learn2learn as l2l
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import Dataset
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.utilities.types import _PATH


class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class MAMLCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        checkpoint = self._remove_redundant_prefix_from_weight_names(checkpoint)
        super().save_checkpoint(checkpoint, path, storage_options)

    @staticmethod
    def _remove_redundant_prefix_from_weight_names(checkpoint: Dict[str, Any]):
        """Turn parameter names of the form `maml.module.a.b.c` into `a.b.c` form."""
        new_dict = OrderedDict()
        state_dict = checkpoint["state_dict"]
        while len(state_dict) > 0:
            k, p = state_dict.popitem()
            cmps = k.split(".")
            if k.startswith("maml.module."):
                new_key = ".".join(cmps[2:])
                new_dict[new_key] = p
        checkpoint["state_dict"] = new_dict
        return checkpoint


class PtTaskDataset(Dataset):
    def __init__(self, taskset: l2l.data.TaskDataset, epoch_length: int):
        super().__init__()
        self.taskset = taskset
        self.epoch_length = epoch_length

    def __getitem__(self, *args, **kwargs):
        return self.taskset.sample()

    def __len__(self):
        return self.epoch_length
