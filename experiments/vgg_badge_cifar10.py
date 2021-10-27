import argparse
import random
from copy import deepcopy
from typing import List

import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.modelwrapper import ModelWrapper

"""
Minimal example to use BaaL with BADGE.
NOTES:
    We don't use MCD for this example. Therefore, iterations is set to 1.
    n_data_to_label here represents the number of centroids for KMEANS++ in BADGE.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--n_data_to_label", default=2, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="badge", type=str)
    parser.add_argument("--iterations", default=1, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument('--learning_epoch', default=20, type=int)
    return parser.parse_args()

from torch.utils.data import Dataset
import numpy as np
class DummyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 20

    def __getitem__(self, item):
        x = torch.from_numpy(np.ones([3, 10, 10]) * item / 255.).float()
        if self.transform:
            x = self.transform(x)
        return x, torch.LongTensor([item % 2])


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
    train_ds = datasets.CIFAR10('/app/cifar/', train=True,
                                transform=transform, target_transform=None, download=False)
    test_set = datasets.CIFAR10('/app/cifar', train=False,
                                transform=test_transform, target_transform=None, download=False)

    active_set = ActiveLearningDataset(train_ds, pool_specifics={'transform': test_transform})

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set

def fake_dataset(initial_pool):
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    train_set = DummyDataset(transform=transform)
    test_set = DummyDataset(transform=transform)
    active_set = ActiveLearningDataset(train_set)
    active_set.label_randomly(2)
    return active_set, test_set


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)

    # active_set, test_set = get_datasets(hyperparams['initial_pool'])
    active_set, test_set = fake_dataset(hyperparams['initial_pool'])

    heuristic = get_heuristic(hyperparams['heuristic'],
                              hyperparams['shuffle_prop'],
                              n_centroids=hyperparams['n_data_to_label'])

    criterion = CrossEntropyLoss()
    model = vgg16(pretrained=False, num_classes=10)
    # weights = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth')
    # weights = {k: v for k, v in weights.items() if 'classifier.6' not in k}
    # model.load_state_dict(weights, strict=False)
    # print(list(model.named_modules()))

    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    # Wraps the model into a usable API.
    model = ModelWrapper(model, criterion, embedding_layer="classifier.6")

    logs = {}
    logs['epoch'] = 0

    # for prediction we use a smaller batchsize
    # since it is slower
    active_loop = ActiveLearningLoop(active_set,
                                     model.get_embedding_grads,
                                     heuristic,
                                     hyperparams.get('n_data_to_label', 1),
                                     optimizer=optimizer,
                                     batch_size=10,
                                     use_cuda=use_cuda)
    # We will reset the weights at each active learning step.
    init_weights = deepcopy(model.state_dict())

    for epoch in tqdm(range(args.epoch)):
        # Load the initial weights.
        model.load_state_dict(init_weights)
        # model.train_on_dataset(active_set, optimizer, hyperparams["batch_size"], hyperparams['learning_epoch'],
        #                        use_cuda)
        #
        # # Validation!
        # model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda)
        # metrics = model.metrics
        should_continue = active_loop.step()
        if not should_continue:
            break
        #
        # val_loss = metrics['test_loss'].value
        # logs = {
        #     "val": val_loss,
        #     "epoch": epoch,
        #     "train": metrics['train_loss'].value,
        #     "labeled_data": active_set.labelled,
        #     "Next Training set size": len(active_set)
        # }
        # print(logs)


if __name__ == "__main__":
    main()
