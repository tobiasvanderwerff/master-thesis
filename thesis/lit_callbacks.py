import re

import matplotlib.pyplot as plt
from pytorch_lightning import Callback


class LogLearnableInnerLoopLearningRates(Callback):
    """Logs the learnable inner loop learning rates used for MAML, in a bar plot."""

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        # Collect all inner loop learning rates.
        lrs = []
        for n, p in pl_module.state_dict().items():
            if n.startswith("model.gbml.compute_update"):
                ix = int(re.search(r"[0-9]+", n).group(0))
                lrs.append((ix, p.item()))
        assert lrs != []

        # Plot the learning rates.
        xs, ys = zip(*lrs)
        fig = plt.figure()
        plt.bar(xs, ys, align="edge", alpha=0.5)
        plt.grid(True)
        plt.ylabel("learning rate")

        # Log to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(f"inner loop learning rates", fig, trainer.global_step)
        plt.close(fig)
