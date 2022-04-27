"""Plot writer codes in the form of Hinge features."""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from thesis.writer_code.util import CODE_TYPES, load_hinge_codes


def plot_pca(codes: np.ndarray):
    pca = PCA(n_components=2)
    codes_pca = pca.fit_transform(codes)
    variance_explained = sum(pca.explained_variance_ratio_) * 100

    plt.figure()
    plt.scatter(codes_pca[:, 0], codes_pca[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"PCA writer codes (retained variance: {variance_explained:.2f}%)")
    plt.show()


def plot_tsne(codes: np.ndarray):
    tsne = TSNE(
        n_components=2, perplexity=25, init="pca", learning_rate="auto", verbose=1
    )
    # I am not sure how the perplexity parameter should be set here. Since the number
    # of codes is not that large (657), I set it to a lower value than what is
    # default (30).
    codes_tsne = tsne.fit_transform(codes)

    plt.figure()
    plt.scatter(codes_tsne[:, 0], codes_tsne[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"t-SNE Writer codes")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        choices=CODE_TYPES,
        help="Hinge feature to plot",
    )
    parser.add_argument("--plot_type", type=str, choices=["pca", "tsne"], default="pca")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    wrt2code, _ = load_hinge_codes(root, args.feature, normalize=False)
    codes = np.stack(list(wrt2code.values()), 0)

    print(f"Plotting {len(codes)} codes.")

    if args.plot_type == "pca":
        plot_pca(codes)
    else:
        plot_tsne(codes)
