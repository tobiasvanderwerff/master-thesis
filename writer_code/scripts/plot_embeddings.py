"""Plot writer codes (embeddings)."""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embs_pca(embs: np.ndarray):
    pca = PCA(n_components=2)
    embs_pca = pca.fit_transform(embs)
    variance_explained = sum(pca.explained_variance_ratio_) * 100

    plt.figure()
    plt.scatter(embs_pca[:, 0], embs_pca[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Writer embeddings (retained variance: {variance_explained:.2f}%)")
    plt.show()


def plot_embs_tsne(embs: np.ndarray):
    pass  # TODO


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_path", type=str, required=True,
    #                     help="Path to model checkpoint containing the embeddings")
    # args = parser.parse_args()

    checkpoint_path = "/home/tobias/Dropbox/master_AI/thesis/code/writer_code/lightning_logs/WriterCodeAdaptiveModel-wer=0.1227=5_lr=1e-4_shots=8_seed=3/checkpoints/WriterCodeAdaptiveModel-epoch=57-char_error_rate=0.0640-word_error_rate=0.0913.ckpt"

    assert Path(
        checkpoint_path
    ).is_file(), f"{checkpoint_path} does not point to a valid file."

    cpt = torch.load(checkpoint_path, map_location="cpu")
    embeddings = cpt["state_dict"]["writer_embs.weight"].numpy()
    print(f"Plotting {embeddings.shape[0]} embeddings.")
    plot_embs_pca(embeddings)
