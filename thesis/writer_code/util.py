from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

AVAILABLE_MODELS = [
    "LitWriterCodeAdaptiveModel",
    "LitWriterCodeAdaptiveModelNonEpisodic",
]

CODE_TYPES = [
    "hinge",
    "quadhinge",
    "cohinge",
    "cochaincode-hinge",
    "triplechaincode-hinge",
    "delta-hinge",
]

ADAPTATION_METHODS = ["conditional_batchnorm", "cnn_output"]


class AdaptationMethod(Enum):
    CONDITIONAL_BATCHNORM = 1
    CNN_OUTPUT = 2
    NONE = 3

    @staticmethod
    def from_string(s: str):
        s = s.lower()
        if s == "conditional_batchnorm":
            return AdaptationMethod.CONDITIONAL_BATCHNORM
        elif s == "cnn_output":
            return AdaptationMethod.CNN_OUTPUT
        elif s == "none":
            return AdaptationMethod.NONE
        else:
            raise ValueError(f"{s} is not a valid adaptation method.")


class WriterEmbeddingType(Enum):
    """
    Type of writer embedding. Choices:

    learned: an embedding whose weights are learned with backpropagation
    transformed: an embedding produced by some transform of features, e.g. a feature
        map passed through a dense layer to produce the embedding.
    """

    LEARNED = 1
    TRANSFORMED = 2

    @staticmethod
    def from_string(s: str):
        s = s.lower()
        if s == "learned":
            return WriterEmbeddingType.LEARNED
        elif s == "transformed":
            return WriterEmbeddingType.TRANSFORMED
        else:
            raise ValueError(f"{s} is not a valid embedding method.")


def load_hinge_codes(
    root_path: Path, code_name: str = "hinge", normalize: bool = True
) -> Tuple[Dict[str, np.array], int]:
    """Load hinge features per writer. Other variants of Hinge can also be loaded, e.g.  Quadhinge.

    Args:
        root_path (Path): thesis root path
        normalize (bool): whether to normalize the features
        code_name (str): what type of features to use

    Returns:
        - dictionary mapping writer identity to writer code
        - code size
    """
    assert code_name in CODE_TYPES, f"{code_name} is not a valid feature name."

    hinge_path = root_path / f"hinge-feature-extraction/{code_name}"
    doc_info = (root_path / "iam_form_to_writerid.txt").read_text()
    docid_to_writerid = {
        line.split()[0]: line.split()[1] for line in doc_info.splitlines()
    }

    writer_to_code = dict()
    for pth in hinge_path.iterdir():
        line = pth.read_text()
        features = line.split()[1:]
        code = np.array([float(ft) for ft in features], dtype=np.float32)
        writer = int(docid_to_writerid[pth.stem])
        # Note that in IAM, one writer can correspond to several docs, i.e. one
        # writer may have hinge features for several docs. For simplicity,
        # we simply overwrite the writer code in case it already exists.
        writer_to_code.update({writer: code})

    if normalize:
        # Normalize features to have 0 mean and stddev 1.
        all_features = np.concatenate(list(writer_to_code.values()))
        mean, std = all_features.mean(), all_features.std()
        writer_to_code = {
            writer: (code - mean) / std for writer, code in writer_to_code.items()
        }

    code_size = list(writer_to_code.values())[0].size

    return writer_to_code, code_size
