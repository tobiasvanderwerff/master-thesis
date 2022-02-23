from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any

from thesis.util import TrainMode

import learn2learn as l2l
from torch import Tensor


class MAMLLearner(ABC):
    @property
    @abstractmethod
    def meta_weights(self):
        """
        Names of the weights specific to MAML, e.g. the names of inner loop
        learnable learning rate weights. Will be used to load meta-weights
        from a saved checkpoint.
        """
        pass

    @abstractmethod
    def meta_learn(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.TRAIN
    ) -> Tuple[Tensor, Tensor, Optional[Dict[int, List]]]:
        """Process a single batch of tasks."""
        pass

    @abstractmethod
    def fast_adaptation(
        self,
        learner: l2l.algorithms.GBML,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
    ) -> Tuple[Any, float, Optional[Tensor]]:
        """Take a single gradient step on a batch of data."""
        pass

    @abstractmethod
    def forward(
        self,
        adaptation_imgs: Tensor,
        adaptation_targets: Tensor,
        inference_imgs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pass
