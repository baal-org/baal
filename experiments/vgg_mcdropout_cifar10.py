import argparse
from pprint import pprint
import random
from copy import deepcopy

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
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper

"""
Minimal example to use BaaL.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=20, type=int)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    # Note: We use the test set here as an example. You should make your own validation set.
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )
    test_set = datasets.CIFAR10(
        ".", train=False, transform=test_transform, target_transform=None, download=True
    )

    active_set = ActiveLearningDataset(train_ds, pool_specifics={"transform": test_transform})

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
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

    active_set, test_set = get_datasets(hyperparams["initial_pool"])

    heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
    criterion = CrossEntropyLoss()
    model = vgg16(pretrained=False, num_classes=10)
    weights = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth")
    weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
    model.load_state_dict(weights, strict=False)

    # change dropout layer to MCDropout
    model = patch_module(model)

    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    # Wraps the model into a usable API.
    model = ModelWrapper(model, criterion)

    logs = {}
    logs["epoch"] = 0

    # for prediction we use a smaller batchsize
    # since it is slower
    active_loop = ActiveLearningLoop(
        active_set,
        model.predict_on_dataset,
        heuristic,
        hyperparams.get("query_size", 1),
        batch_size=10,
        iterations=hyperparams["iterations"],
        use_cuda=use_cuda,
    )
    # We will reset the weights at each active learning step.
    init_weights = deepcopy(model.state_dict())

    for _ in tqdm(range(args.epoch)):
        # Load the initial weights.
        model.load_state_dict(init_weights)
        model.train_on_dataset(
            active_set,
            optimizer,
            hyperparams["batch_size"],
            hyperparams["learning_epoch"],
            use_cuda,
        )

        # Validation!
        model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda)
        should_continue = active_loop.step()
        if not should_continue:
            break

        pprint(model.get_metrics())


if __name__ == "__main__":
    main()
