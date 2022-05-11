"""Plot writer codes (embeddings)."""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embs_pca(embs: np.ndarray):
    pca = PCA(n_components=2)
    embs_pca = pca.fit_transform(embs)
    variance_explained = sum(pca.explained_variance_ratio_) * 100

    plt.figure()
    sns.scatterplot(embs_pca[:, 0], embs_pca[:, 1])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"PCA writer codes (retained variance: {variance_explained:.2f}%)")
    plt.show()


def plot_embs_tsne(embs: np.ndarray):
    tsne = TSNE(n_components=2, perplexity=10, verbose=1)
    embs_tsne = tsne.fit_transform(embs)

    # plt.figure()
    # plt.scatter(embs_tsne[:, 0], embs_tsne[:, 1])
    sns.scatterplot(x=embs_tsne[:, 0], y=embs_tsne[:, 1])

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Learned writer codes")
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_path", type=str, required=True,
    #                     help="Path to model checkpoint containing the embeddings")
    # args = parser.parse_args()

    checkpoint_path = "/tmp/tmp.b1LiY3I2wn/WriterCodeAdaptiveModel-fphtr18_conditional_batchnorm_lr=1e-3_lr-emb=1e-3_adapt-opt-steps=1_num-hidden=128_shots=16_seed=1/checkpoints/WriterCodeAdaptiveModel-epoch=15-char_error_rate=0.1544-word_error_rate=0.1925.ckpt"

    assert Path(
        checkpoint_path
    ).is_file(), f"{checkpoint_path} does not point to a valid file."

    sns.set_theme()

    cpt = torch.load(checkpoint_path, map_location="cpu")
    embeddings = cpt["state_dict"]["model.writer_embs.weight"].numpy()
    print(f"Plotting {embeddings.shape[0]} embeddings.")

    # plot_embs_pca(embeddings)
    plot_embs_tsne(embeddings)
