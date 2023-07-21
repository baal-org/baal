from copy import deepcopy
from pprint import pprint

import torch.cuda
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import MNIST

from baal import ActiveLearningDataset, ModelWrapper
from baal.active import ActiveLearningLoop
from baal.active.heuristics import EPIG
from baal.bayesian.dropout import patch_module

use_cuda = torch.cuda.is_available()

train_transform = transforms.Compose([transforms.RandomRotation(30), transforms.ToTensor()])
test_transform = transforms.ToTensor()
train_ds = MNIST("/tmp", train=True, transform=train_transform, download=True)
test_ds = MNIST("/tmp", train=False, transform=test_transform, download=True)

# Uses an ActiveLearningDataset to help us split labelled and unlabelled examples.
al_dataset = ActiveLearningDataset(train_ds, pool_specifics={"transform": test_transform})
al_dataset.label_randomly(200)  # Start with 200 items labelled.

# Creates an MLP to classify MNIST
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.Dropout(),
    nn.Linear(512, 512),
    nn.Dropout(),
    nn.Linear(512, 10),
)
model = patch_module(model)  # Set dropout layers for MC-Dropout.
if use_cuda:
    model = model.cuda()
wrapper = ModelWrapper(model=model, criterion=nn.CrossEntropyLoss())
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# We will use BALD as our heuristic as it is a great tradeoff between performance and efficiency.
epig = EPIG()
# Setup our active learning loop for our experiments
al_loop = ActiveLearningLoop(
    dataset=al_dataset,
    get_probabilities=wrapper.predict_on_dataset,
    heuristic=epig,
    query_size=100,  # We will label 100 examples per step.
    # KWARGS for predict_on_dataset
    iterations=20,  # 20 sampling for MC-Dropout
    batch_size=32,
    use_cuda=use_cuda,
    verbose=False,
)

# Following Gal 2016, we reset the weights at the beginning of each step.
initial_weights = deepcopy(model.state_dict())

for step in range(100):
    model.load_state_dict(initial_weights)
    train_loss = wrapper.train_on_dataset(
        al_dataset, optimizer=optimizer, batch_size=32, epoch=10, use_cuda=use_cuda
    )
    test_loss = wrapper.test_on_dataset(test_ds, batch_size=32, use_cuda=use_cuda)

    pprint(wrapper.get_metrics())
    flag = al_loop.step()
    if not flag:
        # We are done labelling! stopping
        break
