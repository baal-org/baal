import dataclasses
import math
import unittest
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchmetrics
from torch import nn
from torch.utils.data import Dataset, DataLoader

from baal.modelwrapper import ModelWrapper, mc_inference, TrainingArgs
from baal.utils.metrics import ClassificationReport


class DummyDataset(Dataset):
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


class ModelWrapperMultiOutMultiInTest(unittest.TestCase):
    def setUp(self):
        class MultiOutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleModel()

            def forward(self, x):
                return [self.model(x)] * 2

        self._crit = nn.MSELoss()
        self.criterion = lambda x, y: self._crit(x[0], y) + self._crit(x[1], y)
        self.model = MultiOutModel()
        self.optim = torch.optim.SGD(self.model.parameters(), 0.01)
        self.dataset = DummyDataset()
        self.args = TrainingArgs(criterion=self.criterion,
                                 optimizer=self.optim,
                                 batch_size=4, epoch=1, use_cuda=False, workers=0)
        self.wrapper = ModelWrapper(self.model, args=self.args)

    def test_train_on_batch(self):
        self.wrapper.train()
        old_param = list(map(lambda x: x.clone(), self.model.parameters()))
        input, target = [torch.stack(v) for v in zip(*(self.dataset[0], self.dataset[1]))]
        self.wrapper.train_on_batch(input, target)
        new_param = list(map(lambda x: x.clone(), self.model.parameters()))
        assert any([not torch.allclose(i, j) for i, j in zip(old_param, new_param)])

    def test_test_on_batch(self):
        self.wrapper.eval()
        input, target = [torch.stack(v) for v in zip(*(self.dataset[0], self.dataset[1]))]
        preds = torch.stack(
            [self.wrapper.test_on_batch(input, target) for _ in range(10)]
        ).view(10, -1)

        # Same loss
        assert torch.allclose(torch.mean(preds, 0), preds[0])

        preds = torch.stack(
            [
                self.wrapper.test_on_batch(
                    input, target, average_predictions=10
                )
                for _ in range(10)
            ]
        ).view(10, -1)
        assert torch.allclose(torch.mean(preds, 0), preds[0])

    def test_predict_on_batch(self):
        self.wrapper.eval()
        input = torch.stack((self.dataset[0][0], self.dataset[1][0]))

        # iteration == 1
        pred = self.wrapper.predict_on_batch(input, iterations=1)
        assert pred[0].size() == (2, 1, 1)

        # iterations > 1
        pred = self.wrapper.predict_on_batch(input, 10, )
        assert pred[0].size() == (2, 1, 10)

        # iteration == 1
        new_args = dataclasses.replace(self.args)
        new_args.replicate_in_memory = False
        self.wrapper = ModelWrapper(self.model, new_args)
        pred = self.wrapper.predict_on_batch(input, 1)
        assert pred[0].size() == (2, 1, 1)

        # iterations > 1
        pred = self.wrapper.predict_on_batch(input, 10)
        assert pred[0].size() == (2, 1, 10)

    def test_out_of_mem_raises_error(self):
        self.wrapper.eval()
        input = torch.stack((self.dataset[0][0], self.dataset[1][0]))
        with pytest.raises(RuntimeError) as e_info:
            self.wrapper.predict_on_batch(input, 0)
        assert 'CUDA ran out of memory while BaaL tried to replicate data' in str(e_info.value)

    def test_raising_type_errors(self):
        iterations = math.inf
        self.wrapper.eval()
        input = torch.stack((self.dataset[0][0], self.dataset[1][0]))
        with pytest.raises(TypeError):
            self.wrapper.predict_on_batch(input, iterations)

    def test_train(self):
        new_args = dataclasses.replace(self.args)
        new_args.epoch = 2
        wrapper = ModelWrapper(model=self.model, args=new_args)
        history = wrapper.train_on_dataset(self.dataset)
        assert len(history) == 2

    def test_test(self):
        l = self.wrapper.test_on_dataset(self.dataset, 10)
        assert np.isfinite(l)
        l = self.wrapper.test_on_dataset(
            self.dataset, average_predictions=10
        )
        assert np.isfinite(l)

    def test_predict(self):
        l = self.wrapper.predict_on_dataset(self.dataset, 20,)
        self.wrapper.eval()
        assert np.allclose(
            self.wrapper.predict_on_batch(self.dataset[0][0].unsqueeze(0), 20)[0].detach().numpy(),
            l[0][0])
        assert np.allclose(
            self.wrapper.predict_on_batch(self.dataset[19][0].unsqueeze(0), 20)[0][
                0].detach().numpy(),
            l[0][19])
        assert l[0].shape == (len(self.dataset), 1, 20)

        # Test generators
        l_gen = self.wrapper.predict_on_dataset_generator(self.dataset, 20)
        assert np.allclose(next(l_gen)[0][0], l[0][0])
        for last in l_gen:
            pass  # Get last item
        assert np.allclose(last[0][-1], l[0][-1])

        # Test Half
        l_gen = self.wrapper.predict_on_dataset_generator(self.dataset, 20,  half=True)
        l = self.wrapper.predict_on_dataset(self.dataset, 10,
                                            half=True)
        assert next(l_gen)[0].dtype == np.float16
        assert l[0].dtype == np.float16

        data_s = []
        l_gen = self.wrapper.predict_on_dataset_generator(data_s, 20, half=True)

        assert len(list(l_gen)) == 0


class ModelWrapperTest(unittest.TestCase):
    def setUp(self):
        # self.model = nn.Sequential(
        #     nn.Linear(10, 8), nn.ReLU(), nn.Dropout(), nn.Linear(8, 1), nn.Sigmoid()
        # )
        self.model = SimpleModel()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), 0.01)
        self.dataset = DummyDataset()
        self.args = TrainingArgs(criterion=self.criterion,
                                 optimizer=self.optim,
                                 batch_size=4, epoch=2, use_cuda=False, workers=0)
        self.wrapper = ModelWrapper(self.model, args=self.args)

    def test_train_on_batch(self):
        self.wrapper.train()
        old_param = list(map(lambda x: x.clone(), self.model.parameters()))
        input, target = torch.randn([1, 3, 10, 10]), torch.randn(1, 1)
        self.wrapper.train_on_batch(input, target)
        new_param = list(map(lambda x: x.clone(), self.model.parameters()))

        assert any([not torch.allclose(i, j) for i, j in zip(old_param, new_param)])

        # test reset weights properties
        linear_weights = list(self.wrapper.model.named_children())[3][1].weight.clone()
        conv_weights = list(self.wrapper.model.named_children())[0][1].weight.clone()
        self.wrapper.reset_fcs()
        linear_new_weights = list(self.wrapper.model.named_children())[3][1].weight.clone()
        conv_new_weights = list(self.wrapper.model.named_children())[0][1].weight.clone()
        assert all([not torch.allclose(i, j) for i, j in zip(linear_new_weights, linear_weights)])
        assert all([torch.allclose(i, j) for i, j in zip(conv_new_weights, conv_weights)])

        self.wrapper.reset_all()
        conv_next_new_weights = list(self.wrapper.model.named_children())[0][1].weight.clone()
        assert all(
            [not torch.allclose(i, j) for i, j in zip(conv_new_weights, conv_next_new_weights)])

    def test_test_on_batch(self):
        self.wrapper.eval()
        input, target = torch.randn([1, 3, 10, 10]), torch.randn(1, 1)
        preds = torch.stack(
            [self.wrapper.test_on_batch(input, target) for _ in range(10)]
        ).view(10, -1)

        # Same loss
        assert torch.allclose(torch.mean(preds, 0), preds[0])

        preds = torch.stack(
            [
                self.wrapper.test_on_batch(
                    input, target, average_predictions=10
                )
                for _ in range(10)
            ]
        ).view(10, -1)
        assert torch.allclose(torch.mean(preds, 0), preds[0])

    def test_predict_on_batch(self):
        self.wrapper.eval()
        input = torch.randn([2, 3, 10, 10])

        # iteration == 1
        pred = self.wrapper.predict_on_batch(input, 1,)
        assert pred.size() == (2, 1, 1)

        # iterations > 1
        pred = self.wrapper.predict_on_batch(input, 10,)
        assert pred.size() == (2, 1, 10)

        # iteration == 1
        new_args = dataclasses.replace(self.args)
        new_args.replicate_in_memory = False
        wrapper = ModelWrapper(self.model, new_args)
        pred = wrapper.predict_on_batch(input, 1)
        assert pred.size() == (2, 1, 1)

        # iterations > 1
        pred = wrapper.predict_on_batch(input, 10)
        assert pred.size() == (2, 1, 10)

    def test_train(self):
        history = self.wrapper.train_on_dataset(self.dataset)
        assert len(history) == 2

    def test_test(self):
        l = self.wrapper.test_on_dataset(self.dataset, 10)
        assert np.isfinite(l)
        l = self.wrapper.test_on_dataset(
            self.dataset,  average_predictions=10
        )
        assert np.isfinite(l)

    def test_predict(self):
        l = self.wrapper.predict_on_dataset(self.dataset, 20, )
        self.wrapper.eval()
        assert np.allclose(
            self.wrapper.predict_on_batch(self.dataset[0][0].unsqueeze(0), 20)[0].detach().numpy(),
            l[0])
        assert np.allclose(
            self.wrapper.predict_on_batch(self.dataset[19][0].unsqueeze(0), 20)[0].detach().numpy(),
            l[19])
        assert l.shape == (len(self.dataset), 1, 20)

        # Test generators
        l_gen = self.wrapper.predict_on_dataset_generator(self.dataset, 20, )
        assert np.allclose(next(l_gen)[0], l[0])
        for last in l_gen:
            pass  # Get last item
        assert np.allclose(last[-1], l[-1])

        # Test Half
        l_gen = self.wrapper.predict_on_dataset_generator(self.dataset, 20, half=True)
        l = self.wrapper.predict_on_dataset(self.dataset, 20,
                                            half=True)
        assert next(l_gen).dtype == np.float16
        assert l.dtype == np.float16

    def test_states(self):
        input = torch.randn([1, 3, 10, 10])

        def pred_with_dropout(replicate_in_memory):
            new_args = dataclasses.replace(self.args)
            new_args.replicate_in_memory = replicate_in_memory
            wrapper = ModelWrapper(self.model, new_args)
            wrapper.train()
            # Dropout make the pred changes
            preds = torch.stack(
                [
                    wrapper.predict_on_batch(input, iterations=1)
                    for _ in range(10)
                ]
            ).view(10, -1)
            assert not torch.allclose(torch.mean(preds, 0), preds[0])

        pred_with_dropout(replicate_in_memory=True)
        pred_with_dropout(replicate_in_memory=False)

        def pred_without_dropout(replicate_in_memory):
            new_args = dataclasses.replace(self.args)
            new_args.replicate_in_memory = replicate_in_memory
            wrapper = ModelWrapper(self.model, new_args)
            # Dropout is not active in eval
            wrapper.eval()
            preds = torch.stack(
                [
                    wrapper.predict_on_batch(input, iterations=1)
                    for _ in range(10)
                ]
            ).view(10, -1)
            assert torch.allclose(torch.mean(preds, 0), preds[0])

        pred_without_dropout(replicate_in_memory=True)
        pred_without_dropout(replicate_in_memory=False)

    def test_add_metric(self):
        self.wrapper.add_metric('cls_report', lambda: ClassificationReport(2))
        assert 'test_cls_report' in self.wrapper.metrics
        assert 'train_cls_report' in self.wrapper.metrics
        self.wrapper.train_on_dataset(self.dataset)
        self.wrapper.test_on_dataset(self.dataset, )
        assert (self.wrapper.metrics['train_cls_report'].value['accuracy'] != 0).any()
        assert (self.wrapper.metrics['test_cls_report'].value['accuracy'] != 0).any()

    def test_train_and_test(self):
        res = self.wrapper.train_and_test_on_datasets(self.dataset, self.dataset,
                                                      return_best_weights=False)
        assert len(res) == 2
        res = self.wrapper.train_and_test_on_datasets(self.dataset, self.dataset,   return_best_weights=True)
        assert len(res) == 2
        assert len(res[0]) == 2
        assert isinstance(res[1], dict)
        mock = Mock()
        mock.side_effect = (((np.linspace(0, 50) - 10) / 10) ** 2).tolist()
        new_args = dataclasses.replace(self.args)
        new_args.epoch = 50
        wrapper = ModelWrapper(self.wrapper.model, new_args)
        wrapper.test_on_dataset = mock
        res = wrapper.train_and_test_on_datasets(self.dataset, self.dataset, return_best_weights=True, patience=1)

        assert len(res) == 2
        assert len(res[0]) < 50

        mock = Mock()
        mock.side_effect = (((np.linspace(0, 50) - 10) / 10) ** 2).tolist()

        # iteration == 1
        new_args = dataclasses.replace(self.args)
        new_args.epoch = 50
        wrapper = ModelWrapper(self.wrapper.model, new_args)
        wrapper.test_on_dataset = mock
        res = wrapper.train_and_test_on_datasets(self.dataset, self.dataset, return_best_weights=True, patience=1,
                                                      min_epoch_for_es=20)
        assert len(res) == 2
        assert len(res[0]) < 50 and len(res[0]) > 20

    def test_torchmetric(self):
        mse_fn = lambda: torchmetrics.MeanSquaredError()
        corr_fn = lambda: torchmetrics.SpearmanCorrCoef()
        wrapper = ModelWrapper(self.model, self.args)
        wrapper.add_metric('mse', mse_fn)
        wrapper.add_metric('corr', corr_fn)
        wrapper.train_on_dataset(self.dataset)
        wrapper.test_on_dataset(self.dataset)
        metrics = wrapper.get_metrics()
        assert {'train_corr', 'test_corr', 'train_mse', 'test_mse'}.issubset(metrics.keys())  # Torchmetric metric
        assert {'train_loss', 'test_loss'}.issubset(metrics.keys())  # Baal metric


def test_multi_input_model():
    class MultiInModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleModel()

        def forward(self, x):
            # We get two inputs
            x1, x2 = x
            # We merge those inputs
            return self.model(x1) + self.model(x2)

    model = MultiInModel()
    wrapper = ModelWrapper(model, TrainingArgs(batch_size=15, epoch=1, use_cuda=False, optimizer=None))
    dataset = DummyDataset(n_in=2)
    assert len(dataset[0]) == 2
    b = next(iter(DataLoader(dataset, 15, False)))[0]
    l = wrapper.predict_on_batch(b, iterations=10)
    assert l.shape[0] == 15 and l.shape[-1] == 10


if __name__ == '__main__':
    pytest.main()
