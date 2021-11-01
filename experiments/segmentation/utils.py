from typing import List

import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision.transforms import transforms

from baal import ActiveLearningDataset

pascal_voc_ids = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
)


def active_pascal(
    path="/tmp",
    *args,
    transform=transforms.ToTensor(),
    test_transform=transforms.ToTensor(),
    **kwargs,
):
    """Get active Pascal-VOC 2102 datasets.
    Arguments:
        path : str
            The root folder for the Pascal dataset
    Returns:
        ActiveLearningDataset
            the active learning dataset, training data
        Dataset
            the evaluation dataset
    """

    return (
        ActiveLearningDataset(
            datasets.VOCSegmentation(
                path, image_set="train", transform=transform, download=False, *args, **kwargs
            )
        ),
        datasets.VOCSegmentation(
            path, image_set="val", transform=test_transform, download=False, *args, **kwargs
        ),
    )


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        dropout = nn.Dropout2d(0.5)
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(dropout, conv2d, upsampling, activation)


def add_dropout(
    model: smp.Unet,
    decoder_channels: List[int] = (256, 128, 64, 32, 16),
    classes=1,
    activation=None,
):
    seg_head = SegmentationHead(
        in_channels=decoder_channels[-1],
        out_channels=classes,
        activation=activation,
        kernel_size=3,
    )
    model.add_module("segmentation_head", seg_head)
    model.initialize()


class FocalLoss(nn.Module):
    """
    References:
        Author: clcarwin
        Site https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target != 0).type(torch.LongTensor).to(self.alpha.device)
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
