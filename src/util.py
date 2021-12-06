import xml.etree.ElementTree as ET
import random
import math
from pathlib import Path
from typing import Union, Tuple, Sequence, Any

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


def identity_collate_fn(x: Sequence[Any]):
    """
    This function can be used for PyTorch dataloaders that return batches of size
    1 and do not require any collation of samples in the batch. This is useful if a
    batch of data is already prepared when it is passed to the dataloader.
    """
    assert len(x) == 1
    return x[0]


def read_xml(xml_file: Union[Path, str]) -> ET.Element:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root


def find_child_by_tag(
    xml_el: ET.Element, tag: str, value: str
) -> Union[ET.Element, None]:
    for child in xml_el:
        if child.get(tag) == value:
            return child
    return None


def randomly_displace_and_pad(
    img: np.ndarray, padded_size: Tuple[int, int], **kwargs
) -> np.ndarray:
    """
    Randomly displace an image within a frame, and pad zeros around the image.

    Args:
        img (np.ndarray): image to process
        padded_size (Tuple[int, int]): (height, width) tuple indicating the size of the frame
    """
    h, w = padded_size
    img_h, img_w = img.shape
    assert (
        h >= img_h and w >= img_w
    ), f"Frame is smaller than the image: ({h}, {w}) vs. ({img_h}, {img_w})"
    res = np.zeros((h, w), dtype=img.dtype)

    pad_top = random.randint(0, h - img_h)
    pad_bottom = pad_top + img_h
    pad_left = random.randint(0, w - img_w)
    pad_right = pad_left + img_w

    res[pad_top:pad_bottom, pad_left:pad_right] = img
    return res


def dpi_adjusting(img: np.ndarray, scale: int, **kwargs) -> np.ndarray:
    height, width = img.shape[:2]
    new_height, new_width = math.ceil(height * scale), math.ceil(width * scale)
    return cv2.resize(img, (new_width, new_height))


def matplotlib_imshow(img: torch.Tensor, one_channel=True):
    assert img.device.type == "cpu"
    if one_channel and img.ndim == 3:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
