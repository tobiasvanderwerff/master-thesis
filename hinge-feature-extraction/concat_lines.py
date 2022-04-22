"""Concatenate line images horizontally to create a single image."""

import argparse
from pathlib import Path

import numpy as np
import cv2 as cv

# This script expects images to be in a folder with one directory for each writer, e.g.
# `img/writer1/`, `img/writer2/`, etc.


def main(img_dir: Path, out_dir: Path):
    out_dir.mkdir(
        exist_ok=True
    )  # NOTE: make sure this folder is not in the image folder!
    for writer_dir in img_dir.iterdir():
        # print(f"Processing {len(list(writer_dir.iterdir()))} images from {writer_dir.name}")

        lines = []
        for im_pth in writer_dir.iterdir():
            img = cv.imread(str(im_pth), cv.IMREAD_GRAYSCALE)
            lines.append(img)

        res_h, res_w = sum(im.shape[0] for im in lines), max(
            im.shape[1] for im in lines
        )
        res = np.zeros((res_h, res_w)) + 255
        pos = 0
        for line in lines:
            h, w = line.shape
            res[pos : pos + h, :w] = line
            pos += h
        cv.imwrite(str(out_dir / (writer_dir.name + ".png")), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=Path)
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()

    print("Processing...")
    main(args.img_dir, args.out_dir)
    print("Done.")

    # print the output directory (useful for piping to another program)
    # print(args.out_dir.resolve())
