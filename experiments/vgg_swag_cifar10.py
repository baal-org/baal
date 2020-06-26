import argparse
import random
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.swag import StochasticWeightAveraging
from baal.ensemble import EnsembleModelWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--initial_pool", default=256, type=int)
    parser.add_argument("--heuristic", default='bald', type=str)
    parser.add_argument("--n_data_to_label", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--learning_epoch", default=10, type=int)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(30),
         transforms.ToTensor(),
         transforms.Normalize(3 * [0.5], 3 * [0.5]), ])
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    # Note: We use the test set here as an example. You should make your own validation set.
    train_ds = datasets.CIFAR10('.', train=True,
                                transform=transform, target_transform=None, download=True)
    test_set = datasets.CIFAR10('.', train=False,
                                transform=test_transform, target_transform=None, download=True)

    active_set = ActiveLearningDataset(train_ds, pool_specifics={'transform': test_transform})

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set


def get_model():
    model = vgg16(pretrained=False, num_classes=10)
    weights = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth')
    weights = {k: v for k, v in weights.items() if 'classifier.6' not in k}
    model.load_state_dict(weights, strict=False)
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    assert use_cuda

    seed_all(args.seed)
    batch_size = args.batch_size

    active_set, test_set = get_datasets(args.initial_pool)

    # We randomly create an initial pool of labelled item
    heuristic = get_heuristic(args.heuristic)
    criterion = CrossEntropyLoss()

    model = get_model()

    if use_cuda:
        model.cuda()
    _optimizer = optim.SGD(model.parameters(), lr=args.lr,
                           momentum=0.9, weight_decay=1e-4)
    model = EnsembleModelWrapper(model, criterion)

    active_loop = ActiveLearningLoop(active_set,
                                     model.predict_on_dataset,
                                     heuristic,
                                     args.n_data_to_label,
                                     batch_size=10,
                                     iterations=args.iterations,
                                     use_cuda=use_cuda)

    init_weight = deepcopy(model.state_dict())
    for epoch in tqdm(range(args.epoch)):
        model.clear_checkpoints()
        optimizer = StochasticWeightAveraging(
            _optimizer,
            lr_max=args.lr,
            lr_min=0.0001,
            swa_start=len(active_set) // batch_size // 5,  # .2 epoch burn-in
            swa_freq=len(active_set) // batch_size // 5  # .2 epoch between samples
        )
        model.load_state_dict(init_weight)
        model.train_on_dataset(active_set, optimizer, batch_size, args.learning_epoch,
                               use_cuda)

        for _ in range(args.iterations):
            optimizer.sample()
            optimizer.bn_update(model.model, DataLoader(active_set, batch_size=batch_size,
                                                        num_workers=4))
            model.add_checkpoint()

        # Validation!
        model.test_on_dataset(test_set, batch_size, use_cuda)
        metrics = model.metrics
        active_loop.step()

        val_loss = metrics['test_loss'].value
        logs = {
            "val": val_loss,
            "epoch": epoch,
            "train": metrics['train_loss'].value,
            "labeled_data": active_set._labelled,
            "Next Training set size": len(active_set)
        }
        pprint(logs)


if __name__ == "__main__":
    main()
