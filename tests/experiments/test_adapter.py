import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import Dataset
from transformers import pipeline, TrainingArguments

from baal import ModelWrapper, ActiveLearningDataset
from baal.active.heuristics import BALD
from baal.active.stopping_criteria import LabellingBudgetStoppingCriterion
from baal.experiments.base import ActiveLearningExperiment
from baal.experiments.modelwrapper import ModelWrapperAdapter
from baal.experiments.transformers import TransformersAdapter
from baal.modelwrapper import TrainingArgs
from baal.transformers_trainer_wrapper import BaalTransformersTrainer


class MockHFDataset:
    def __init__(self):
        self.x = torch.randint(0, 5, (10,))
        self.y = torch.randn((2,))
        self.length = 5

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.y}


class MockDataset(Dataset):
    def __init__(self, n_in=1):
        self.n_in = n_in

    def __len__(self):
        return 20

    def __getitem__(self, item):
        x = torch.from_numpy(np.ones([3, 10, 10]) * item / 255.).float()
        if self.n_in > 1:
            x = [x] * self.n_in
        return x, (torch.FloatTensor([item % 2]))


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


@pytest.fixture
def model_wrapper():
    model = SimpleModel()
    return ModelWrapperAdapter(
        wrapper=ModelWrapper(
            model=model,
            args=TrainingArgs(
                optimizer=SGD(model.parameters(), lr=0.1),
                criterion=MSELoss(),
                epoch=2,
                use_cuda=False
            )
        )
    )


@pytest.fixture
def hf_trainer():
    text_classifier = pipeline(
        task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
    )
    return TransformersAdapter(BaalTransformersTrainer(
        model=text_classifier.model,
        args=TrainingArguments('/tmp', no_cuda=True)
    ))


def test_adapter_reset(model_wrapper):
    dataset = MockDataset()
    before_params = list(
        map(lambda x: x.clone(), model_wrapper.wrapper.model.parameters()))
    model_wrapper.train(dataset)
    after_params = list(
        map(lambda x: x.clone(), model_wrapper.wrapper.model.parameters()))
    assert not all([np.array_equal(i.detach(), j.detach())
                    for i, j in zip(before_params, after_params)])
    model_wrapper.reset_weights()
    reset_params = list(
        map(lambda x: x.clone(), model_wrapper.wrapper.model.parameters()))
    assert all([np.allclose(i.detach(), j.detach())
                for i, j in zip(before_params, reset_params)])
    model_wrapper.train(dataset)

    result = model_wrapper.evaluate(dataset, average_predictions=1)
    assert 'test_loss' in result

    predictions = model_wrapper.predict(dataset, iterations=10)
    assert predictions.shape == (len(dataset), 1, 10)


def test_hf_wrapper(hf_trainer):
    dataset = MockHFDataset()
    before_params = list(
        map(lambda x: x.clone(), hf_trainer.wrapper.model.parameters()))
    hf_trainer.train(dataset)
    after_params = list(
        map(lambda x: x.clone(), hf_trainer.wrapper.model.parameters()))
    assert not all([np.array_equal(i.detach(), j.detach())
                    for i, j in zip(before_params, after_params)])
    hf_trainer.reset_weights()
    reset_params = list(
        map(lambda x: x.clone(), hf_trainer.wrapper.model.parameters()))
    assert all([np.allclose(i.detach(), j.detach())
                for i, j in zip(before_params, reset_params)])
    hf_trainer.train(dataset)

    result = hf_trainer.evaluate(dataset, average_predictions=1)
    assert 'eval_loss' in result

    predictions = hf_trainer.predict(dataset, iterations=10)
    assert predictions.shape == (len(dataset), 2, 10)


def test_experiment(model_wrapper):
    al_dataset = ActiveLearningDataset(MockDataset())
    al_dataset.label_randomly(10)
    experiment = ActiveLearningExperiment(
        trainer=model_wrapper.wrapper, al_dataset=al_dataset, eval_dataset=MockDataset(),
        heuristic=BALD(), query_size=5, iterations=10, criterion=None
    )
    assert isinstance(experiment.criterion, LabellingBudgetStoppingCriterion)
    assert isinstance(experiment.adapter, ModelWrapperAdapter)
    output = experiment.start()
    assert len(output) == 2
