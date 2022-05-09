from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

AVAILABLE_MODELS = [
    "LitWriterCodeAdaptiveModel",
]

ADAPTATION_METHODS = ["conditional_batchnorm", "cnn_output"]


class AdaptationMethod(Enum):
    CONDITIONAL_BATCHNORM = 1
    CNN_OUTPUT = 2

    @staticmethod
    def from_string(s: str):
        s = s.lower()
        if s == "conditional_batchnorm":
            return AdaptationMethod.CONDITIONAL_BATCHNORM
        elif s == "cnn_output":
            return AdaptationMethod.CNN_OUTPUT
        else:
            raise ValueError(f"{s} is not a valid adaptation method.")
