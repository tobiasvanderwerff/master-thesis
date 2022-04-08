from argparse import ArgumentParser
from collections import defaultdict
from typing import Sequence, Dict, Any, List

import numpy as np
from pytorch_lightning import Callback

from htr.util import LabelEncoder
from thesis.lit_callbacks import LogLearnableInnerLoopLearningRates
from thesis.lit_models import LitMAMLLearner
from thesis.metahtr.lit_callbacks import (
    LogInstanceSpecificWeights,
)
from thesis.metahtr.models import MetaHTR

import torch.nn as nn


class LitMetaHTR(LitMAMLLearner):
    def __init__(
        self,
        base_model: nn.Module,
        num_clf_weights: int,
        inst_mlp_hidden_size: int = 8,
        initial_inner_lr: float = 0.001,
        use_instance_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_clf_weights = num_clf_weights
        self.inst_mlp_hidden_size = inst_mlp_hidden_size
        self.initial_inner_lr = initial_inner_lr
        self.use_instance_weights = use_instance_weights

        self.model = MetaHTR(
            base_model=base_model,
            num_clf_weights=num_clf_weights,
            inst_mlp_hidden_size=inst_mlp_hidden_size,
            initial_inner_lr=initial_inner_lr,
            use_instance_weights=use_instance_weights,
            **kwargs,
        )

        self.save_hyperparameters(
            "inst_mlp_hidden_size", "initial_inner_lr", "use_instance_weights"
        )

    def training_epoch_end(self, epoch_outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def validation_epoch_end(self, epoch_outputs):
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def test_epoch_end(self, epoch_outputs):
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def aggregate_epoch_instance_weights(
        self, training_epoch_outputs: Sequence[Dict[Any, Any]]
    ):
        char_to_weights, char_to_avg_weight = defaultdict(list), defaultdict(float)
        # Aggregate all instance-specific weights.
        for dct in training_epoch_outputs:
            char_to_ws = dct["char_to_inst_weights"]
            for c_idx, ws in char_to_ws.items():
                char_to_weights[c_idx].extend(ws)
        # Calculate average instance weight per class for logging purposes.
        for c_idx, ws in char_to_weights.items():
            char_to_avg_weight[c_idx] = np.mean(ws)
        self.char_to_avg_inst_weight = char_to_avg_weight

    def add_model_specific_callbacks(
        self, callbacks: List[Callback], label_encoder: LabelEncoder, **kwargs
    ) -> List[Callback]:
        callbacks = super().add_model_specific_callbacks(
            callbacks,
            label_encoder=label_encoder,
            **kwargs,
        )
        callbacks.extend(
            [
                LogLearnableInnerLoopLearningRates(),
                LogInstanceSpecificWeights(label_encoder),
            ]
        )
        return callbacks

    @staticmethod
    def init_with_base_model_from_checkpoint(*args, **kwargs):
        return LitMAMLLearner.init_with_base_model_from_checkpoint(
            maml_arch="metahtr", *args, **kwargs
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MetaHTR")
        parser.add_argument("--inst_mlp_hidden_size", type=int, default=8)
        parser.add_argument("--initial_inner_lr", type=float, default=0.0001)
        parser.add_argument(
            "--use_instance_weights",
            action="store_true",
            default=False,
            help="Use instance-specific weights proposed in the MetaHTR paper.",
        )
        return parent_parser
