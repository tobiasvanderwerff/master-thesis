from collections import defaultdict
from typing import Optional, Callable

import learn2learn as l2l
from torch.utils.data import Dataset

from htr.data import IAMDataset


class PtTaskDataset(Dataset):
    def __init__(self, taskset: l2l.data.TaskDataset, epoch_length: int):
        super().__init__()
        self.taskset = taskset
        self.epoch_length = epoch_length

    def __getitem__(self, *args, **kwargs):
        return self.taskset.sample()

    def __len__(self):
        return self.epoch_length


class WriterDataset(Dataset):
    """Wraps a IAM dataset that returns all examples for a specified writer."""

    def __init__(
        self,
        dataset: IAMDataset,
        collate_fn: Optional[Callable] = None,
        n_augmented_versions: int = 0,
    ):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.n_augmented_versions = n_augmented_versions

        self.writerid_to_img_idxs = defaultdict(list)
        for idx, (_, wid, _) in enumerate(self.dataset):
            self.writerid_to_img_idxs[wid].append(idx)

    def __len__(self):
        # Return total number of writers in the dataset.
        return len(self.writerid_to_img_idxs)

    def __getitem__(self, writerid: int):
        """Return all examples for a given writer.

        Args:
            writerid (int): writer id
            n_augmented_versions (int): produce n augmented versions of the batch of data
        """
        img_idxs = self.writerid_to_img_idxs[writerid]
        data = [self.dataset[i] for i in img_idxs]
        if self.n_augmented_versions > 0:
            # Produce augmented versions of the same batch of data.
            split = self.dataset._split
            # Set image augmentations.
            self.dataset.set_transforms_for_split("train")
            for _ in range(self.n_augmented_versions):
                data.extend([self.dataset[i] for i in img_idxs])
            self.dataset.set_transforms_for_split(split)
        if self.collate_fn is not None:
            imgs, targets, wids = self.collate_fn(data)
            return imgs, targets, wids
        return data
