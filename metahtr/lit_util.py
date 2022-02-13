from typing import Optional, Dict, Any
from collections import OrderedDict

from metahtr.lit_models import MetaHTR

from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.utilities.types import _PATH


class MetaHTRCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        checkpoint = self._add_correct_prefix_for_weight_names(checkpoint)
        super().save_checkpoint(checkpoint, path, storage_options)

    @staticmethod
    def _add_correct_prefix_for_weight_names(checkpoint: Dict[str, Any]):
        """
        Turn parameter names of the form `model.module.a.b.c` into `model.a.b.c`
        form. This makes it possible to load the weights using the
        `LitFullPageHTREncoderDecoder` class.
        """
        new_dict = OrderedDict()
        state_dict = checkpoint["state_dict"]
        while len(state_dict) > 0:
            k, p = state_dict.popitem()
            cmps = k.split(".")
            if k.startswith("model.module."):
                if not any(k.startswith(n) for n in MetaHTR.meta_weights):
                    k = "model." + ".".join(cmps[2:])
            new_dict[k] = p
        assert len(new_dict) != 0, "State dict saved for checkpoint is empty."
        checkpoint["state_dict"] = new_dict
        return checkpoint
