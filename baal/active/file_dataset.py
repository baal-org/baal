import random
import warnings
from typing import Any, Callable, Optional, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from baal.utils.transforms import BaaLTransform


def default_image_load_fn(x):
    return Image.open(x).convert("RGB")


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class FileDataset(Dataset):
    """
    Dataset object that load the files and apply a transformation.

    Args:
        files (List[str]): The files.
        lbls (List[Any]): The labels, -1 indicates that the label is unknown.
        transform (Optional[Callable]): torchvision.transform pipeline.
        target_transform (Optional[Callable]): Function that modifies the target.
        image_load_fn (Optional[Callable]): Function that loads the image, by default uses PIL.
        seed (Optional[int]): Will set a seed before and between DA.
    """

    def __init__(
        self,
        files: List[str],
        lbls: Optional[List[Any]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_load_fn: Optional[Callable] = None,
        seed=None,
    ):
        self.files = files

        if lbls is None:
            self.lbls = [-1] * len(self.files)
        else:
            self.lbls = lbls

        self.transform = transform
        self.target_transform = target_transform
        self.image_load_fn = image_load_fn or default_image_load_fn
        self.seed = seed

    def label(self, idx: int, lbl: Any):
        """
        Label the sample `idx` with `lbl`.

        Args:
            idx (int): The sample index.
            lbl (Any): The label to assign.
        """
        if self.lbls[idx] >= 0:
            warnings.warn(
                "We're modifying the class of the sample {} that we already know : {}.".format(
                    self.files[idx], self.lbls[idx]
                ),
                UserWarning,
            )

        self.lbls[idx] = lbl

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, y = self.files[idx], self.lbls[idx]

        np.random.seed(self.seed)
        batch_seed = np.random.randint(0, 100, 1).item()
        seed_all(batch_seed + idx)

        img = self.image_load_fn(x)
        kwargs = self.get_kwargs(self.transform, image_shape=img.size, idx=idx)

        if self.transform:
            img_t = self.transform(img, **kwargs)
        else:
            img_t = img

        if self.target_transform:
            seed_all(batch_seed + idx)
            kwargs = self.get_kwargs(self.target_transform, image_shape=img.size, idx=idx)
            y = self.target_transform(y, **kwargs)
        return img_t, y

    @staticmethod
    def get_kwargs(transform, **kwargs):
        if isinstance(transform, BaaLTransform):
            t_kwargs = {k: kwargs[k] for k in transform.get_requires()}
        else:
            t_kwargs = {}
        return t_kwargs
