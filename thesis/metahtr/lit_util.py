from typing import Optional, Dict, Any, List
from collections import OrderedDict

from thesis.metahtr.lit_models import MetaHTR

from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.utilities.types import _PATH


class MAMLHTRCheckpointIO(TorchCheckpointIO):
    def __init__(self, base_model_params: List[str]):
        self.base_model_params = base_model_params
        self.new_to_old = None

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        if self.new_to_old is None:
            self._init_new_to_old(checkpoint)
        checkpoint = self._correct_base_model_weight_names(checkpoint)
        super().save_checkpoint(checkpoint, path, storage_options)

    def _init_new_to_old(self, checkpoint: Dict[str, Any]):
        """
        Map newly loaded base model weights, which have an additional prefix
        because they are loaded as part of a new model, to their original weight
        names.
        """
        new_to_old = dict()
        state_dict = checkpoint["state_dict"]
        for new in state_dict.keys():
            for old in self.base_model_params:
                if new.endswith(old):
                    new_to_old[new] = old
                    break
        assert len(new_to_old) == len(self.base_model_params)
        self.new_to_old = new_to_old

    def _correct_base_model_weight_names(self, checkpoint: Dict[str, Any]):
        """
        For all base model weights, save them under the same name as they were
        originally loaded. This makes it easier to load the weights from their
        original classes later on.
        """
        new_dict = OrderedDict()
        state_dict = checkpoint["state_dict"]
        n_corrected = 0
        while len(state_dict) > 0:
            wn, w = state_dict.popitem()
            old = self.new_to_old.get(wn)
            if old is not None:
                wn = old
                n_corrected += 1
            new_dict[wn] = w
        assert n_corrected == len(self.base_model_params), (
            f"Not all base model parameters were found: {n_corrected} found, "
            f"whereas {len(self.base_model_params)} should be present."
        )
        checkpoint["state_dict"] = new_dict
        return checkpoint
