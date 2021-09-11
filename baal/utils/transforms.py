import numpy as np
import torch
from PIL import Image


class BaaLTransform:
    def get_requires(self):
        return []


class BaaLCompose(BaaLTransform):
    def __init__(self, transformations):
        self.tfs = transformations

    def get_requires(self):
        result = []
        for t in self.tfs:
            if isinstance(t, BaaLTransform):
                result += t.get_requires()
        return list(set(result))

    def __call__(self, x, **kwargs):
        for t in self.tfs:
            if isinstance(t, BaaLTransform):
                t_kwargs = {k: kwargs[k] for k in t.get_requires()}
            else:
                t_kwargs = {}
            x = t(x, **t_kwargs)
        return x


class GetCanvas(BaaLTransform):
    """Return an empty canvas made from the image and the original data."""

    def get_requires(self):
        return ["image_shape"]

    def __call__(self, x, image_shape):
        return x, np.zeros(image_shape, dtype=np.float32)


class PILToLongTensor(object):
    """Converts a ``PIL Image`` to a ``torch.LongTensor``.
    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor
    """

    def __init__(self, classes=None):
        self.clss = classes

    def __call__(self, pic):
        """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.
        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``
        Returns:
        A ``torch.LongTensor``.
        """
        if isinstance(pic, Image.Image):
            pic = np.array(pic.convert("RGB"))

        if self.clss is not None:
            img = torch.from_numpy(self.encode_segmap(pic))
        else:
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.long()

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.clss):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask
