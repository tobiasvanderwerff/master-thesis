from enum import Enum


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
