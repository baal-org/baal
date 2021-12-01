from copy import deepcopy
from pprint import pprint

import pandas as pd
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from baal import ActiveLearningDataset, ModelWrapper
from baal.active import ActiveLearningLoop
from baal.active.heuristics import Variance
from baal.bayesian.dropout import patch_module

use_cuda = torch.cuda.is_available()


def weight_init_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight, mode="fan_in", nonlinearity="relu")


# DATASET : https://archive.ics.uci.edu/ml/datasets/
# Physicochemical+Properties+of+Protein+Tertiary+Structure#


class FeatureDataset(Dataset):
    def __init__(self, file_name, split="train", seed=42):
        df = pd.read_csv(file_name)
        x = df.iloc[1:, 1:].values
        y = df.iloc[1:, 0].values

        # feature scaling
        sc = StandardScaler()
        x = sc.fit_transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long).unsqueeze(-1)

        # split
        self.split = split
        train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)

        if self.split == "train":
            self.x = x[train_idx]
            self.y = y[train_idx]
        elif self.split == "test":
            self.x = x[test_idx]
            self.y = y[test_idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# Change the following paths with respect to the location of `CASP.csv`
# You might need to add the path to the list of python system paths.
al_dataset = ActiveLearningDataset(FeatureDataset("./data/CASP.csv"))
test_ds = FeatureDataset("./data/CASP.csv", split="test")

al_dataset.label_randomly(1000)  # Start with 1000 items labelled.

# Creates an MLP to classify MNIST
model = nn.Sequential(
    nn.Flatten(), nn.Linear(9, 16), nn.Dropout(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
)

model = patch_module(model)  # Set dropout layers for MC-Dropout.
model.apply(weight_init_normal)

if use_cuda:
    model = model.cuda()
wrapper = ModelWrapper(model=model, criterion=nn.L1Loss())
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# We will use Variance as our heuristic for regression problems.
variance = Variance()

# Setup our active learning loop for our experiments
al_loop = ActiveLearningLoop(
    dataset=al_dataset,
    get_probabilities=wrapper.predict_on_dataset,
    heuristic=variance,
    query_size=250,  # We will label 20 examples per step.
    # KWARGS for predict_on_dataset
    iterations=20,  # 20 sampling for MC-Dropout
    batch_size=16,
    use_cuda=use_cuda,
    verbose=False,
    workers=0,
)

# Following Gal 2016, we reset the weights at the beginning of each step.
initial_weights = deepcopy(model.state_dict())

for step in range(1000):
    model.load_state_dict(initial_weights)
    train_loss = wrapper.train_on_dataset(
        al_dataset, optimizer=optimizer, batch_size=16, epoch=1000, use_cuda=use_cuda, workers=0
    )
    test_loss = wrapper.test_on_dataset(test_ds, batch_size=16, use_cuda=use_cuda, workers=0)

    pprint(
        {
            "dataset_size": len(al_dataset),
            "train_loss": wrapper.metrics["train_loss"].value,
            "test_loss": wrapper.metrics["test_loss"].value,
        }
    )
    flag = al_loop.step()
    if not flag:
        # We are done labelling! stopping
        break
