import argparse
from copy import deepcopy
from pprint import pprint

import torch.backends
from PIL import Image
from torch import optim
from torchvision.transforms import transforms
from tqdm import tqdm

from baal import get_heuristic, ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule
from baal import ModelWrapper
from baal import ClassificationReport
from baal import PILToLongTensor
from utils import pascal_voc_ids, active_pascal, add_dropout, FocalLoss

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise Exception("This example requires `smp`.\n pip install segmentation_models_pytorch")

import torch
import torch.nn.functional as F
import numpy as np


def mean_regions(n, grid_size=16):
    # Compute the mean uncertainty per regions.
    # [batch_size, W, H]
    n = torch.from_numpy(n[:, None, ...])
    # [Batch_size, 1, grid, grid]
    out = F.adaptive_avg_pool2d(n, grid_size)
    return np.mean(out.view([-1, grid_size**2]).numpy(), -1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--al_step", default=200, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--initial_pool", default=40, type=int)
    parser.add_argument("--query_size", default=20, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="random", type=str)
    parser.add_argument("--reduce", default="sum", type=str)
    parser.add_argument("--data_path", default="/data", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--learning_epoch", default=50, type=int)
    return parser.parse_args()


def get_datasets(initial_pool, path):
    IM_SIZE = 224
    # TODO add better data augmentation scheme.
    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(IM_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(IM_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    target_transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=Image.NEAREST),
            transforms.CenterCrop(IM_SIZE),
            PILToLongTensor(pascal_voc_ids),
        ]
    )
    active_set, test_set = active_pascal(
        path=path,
        transform=transform,
        test_transform=test_transform,
        target_transform=target_transform,
    )
    active_set.label_randomly(initial_pool)
    return active_set, test_set


def main():
    args = parse_args()
    batch_size = args.batch_size
    use_cuda = torch.cuda.is_available()
    hyperparams = vars(args)
    pprint(hyperparams)

    active_set, test_set = get_datasets(hyperparams["initial_pool"], hyperparams["data_path"])

    # We will use the FocalLoss
    criterion = FocalLoss(gamma=2, alpha=0.25)

    # Our model is a simple Unet
    model = smp.Unet(
        encoder_name="resnext50_32x4d",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_batchnorm=False,
        classes=len(pascal_voc_ids),
    )
    # Add a Dropout layerto use MC-Dropout
    add_dropout(model, classes=len(pascal_voc_ids), activation=None)

    # This will enable Dropout at test time.
    model = MCDropoutModule(model)

    # Put everything on GPU.
    if use_cuda:
        model.cuda()

    # Make an optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9, weight_decay=5e-4)
    # Keep a copy of the original weights
    initial_weights = deepcopy(model.state_dict())

    # Add metrics
    model = ModelWrapper(model, criterion)
    model.add_metric("cls_report", lambda: ClassificationReport(len(pascal_voc_ids)))

    # Which heuristic you want to use?
    # We will use our custom reduction function.
    heuristic = get_heuristic(hyperparams["heuristic"], reduction=mean_regions)

    # The ALLoop is in charge of predicting the uncertainty and
    loop = ActiveLearningLoop(
        active_set,
        model.predict_on_dataset_generator,
        heuristic=heuristic,
        query_size=hyperparams["query_size"],
        # Instead of predicting on the entire pool, only a subset is used
        max_sample=1000,
        batch_size=batch_size,
        iterations=hyperparams["iterations"],
        use_cuda=use_cuda,
    )
    acc = []
    for epoch in tqdm(range(args.al_step)):
        # Following Gal et al. 2016, we reset the weights.
        model.load_state_dict(initial_weights)
        # Train 50 epochs before sampling.
        model.train_on_dataset(
            active_set, optimizer, batch_size, hyperparams["learning_epoch"], use_cuda
        )

        # Validation!
        model.test_on_dataset(test_set, batch_size, use_cuda)
        should_continue = loop.step()

        logs = model.get_metrics()
        pprint(logs)
        acc.append(logs)
        if not should_continue:
            break


if __name__ == "__main__":
    main()
