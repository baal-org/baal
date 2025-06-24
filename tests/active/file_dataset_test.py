import os
import time
import random
import tempfile
import unittest

import numpy as np
import pytest
import torch
from PIL import Image
from PIL.Image import NEAREST
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomRotation, Compose, ToTensor, ToPILImage

from baal.active import ActiveLearningDataset
from baal.active.file_dataset import FileDataset, default_image_load_fn
from baal.utils.transforms import BaaLCompose, GetCanvas, PILToLongTensor


class FileDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tmp_dir = tempfile.gettempdir()
        paths = []
        for idx in range(100):
            path = os.path.join(tmp_dir, "{}.png".format(idx))
            Image.fromarray(np.random.randint(0, 100, [10, 10, 3], np.uint8)).save(path)
            paths.append(path)
        cls.paths = paths

    def setUp(self):
        self.lbls = None
        self.transform = Compose([Resize(60), RandomRotation(90), ToTensor()])
        testtransform = Compose([Resize(32), ToTensor()])
        self.dataset = FileDataset(self.paths, self.lbls, transform=self.transform)
        self.lbls = self.generate_labels(len(self.paths), 10)
        self.dataset = FileDataset(self.paths, self.lbls, transform=self.transform)
        self.active = ActiveLearningDataset(self.dataset, labelled=(np.array(self.lbls) != -1), pool_specifics={'transform': testtransform})

    def generate_labels(self, n, init_lbls):
        lbls = [-1] * n
        for i in random.sample(range(n), init_lbls):
            lbls[i] = i % 10
        return lbls

    def test_default_label(self):
        dataset = FileDataset(self.paths)
        assert dataset.lbls == [-1] * len(self.paths)

    def test_labelling(self):
        actually_labelled = [i for i, j in enumerate(self.lbls) if j >= 0]
        actually_not_labelled = [i for i, j in enumerate(self.lbls) if j < 0]
        with pytest.warns(UserWarning):
            self.dataset.label(actually_labelled[0], 1)
        self.dataset.label(actually_not_labelled[0], 1)
        assert sum([1 for i, j in enumerate(self.dataset.lbls) if j >= 0]) == 11

    def test_active_labelling(self):
        assert self.active.can_label
        actually_not_labelled = [i for i, j in enumerate(self.lbls) if j < 0]
        actually_labelled = [i for i, j in enumerate(self.lbls) if j >= 0]

        init_length = len(self.active)
        self.active.label(actually_not_labelled[0], 1)
        assert len(self.active) == init_length + 1

        with pytest.warns(UserWarning):
            self.dataset.label(actually_labelled[0], None)
        assert len(self.active) == init_length + 1

    def test_filedataset_segmentation(self):
        target_trans = Compose([default_image_load_fn,
                                Resize(60), RandomRotation(90), ToTensor()])
        file_dataset = FileDataset(self.paths, self.paths, self.transform, target_trans, seed=1337)
        x, y = file_dataset[0]
        assert np.allclose(x.numpy(), y.numpy())
        out1 = list(DataLoader(file_dataset, batch_size=1, num_workers=3, shuffle=False))
        out2 = list(DataLoader(file_dataset, batch_size=1, num_workers=3, shuffle=False))
        assert all([np.allclose(x1.numpy(), x2.numpy())
                    for (x1, _), (x2, _) in zip(out1, out2)])

        file_dataset = FileDataset(self.paths, self.paths, self.transform, target_trans, seed=None)
        x, y = file_dataset[0]
        assert np.allclose(x.numpy(), y.numpy())
        out1 = list(DataLoader(file_dataset, batch_size=1, num_workers=3, shuffle=False))
        out2 = list(DataLoader(file_dataset, batch_size=1, num_workers=3, shuffle=False))
        assert not all([np.allclose(x1.numpy(), x2.numpy())
                        for (x1, _), (x2, _) in zip(out1, out2)])

    def test_segmentation_pipeline(self):
        class DrawSquare:
            def __init__(self, side):
                self.side = side

            def __call__(self, x, **kwargs):
                x, canvas = x  # x is a [int, ndarray]
                canvas[:self.side, :self.side] = x
                return canvas.astype(np.uint8)

        target_trans = BaaLCompose(
            [GetCanvas(), DrawSquare(3), ToPILImage(mode=None), Resize(60, interpolation=0),
             RandomRotation(10, fill=0.0), PILToLongTensor()])
        file_dataset = FileDataset(self.paths, [1] * len(self.paths), self.transform, target_trans)

        x, y = file_dataset[0]
        assert np.allclose(np.unique(y), [0, 1])
        assert y.shape[1:] == x.shape[1:]


if __name__ == '__main__':
    pytest.main()
