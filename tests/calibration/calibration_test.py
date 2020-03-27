import unittest
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from baal.calibration import DirichletCalibrator
from baal.modelwrapper import ModelWrapper


class DummyDataset(Dataset):

    def __len__(self):
        return 20

    def __getitem__(self, item):
        return torch.from_numpy(np.ones([3, 10, 10]) * item / 255.).float(), \
               (item % 2)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.criterion = nn.CrossEntropyLoss()
        self.wrapper = ModelWrapper(self.model, self.criterion)
        self.optim = torch.optim.SGD(self.wrapper.get_params(), 0.01)
        self.dataset = DummyDataset()
        self.calibrator = DirichletCalibrator(self.wrapper, 2, lr=0.001, reg_factor=0.001)

    def test_calibrated_model(self):
        assert len(list(self.calibrator.init_model.modules())) < len(
            list(self.calibrator.calibrated_model.modules()))

    def test_calibration(self):
        before_calib_param_init = list(map(lambda x: x.clone(), self.calibrator.init_model.parameters()))
        before_calib_param = list(map(lambda x: x.clone(), self.calibrator.calibrated_model.parameters()))

        self.calibrator.calibrate(self.dataset, self.dataset,
                                  batch_size=10, epoch=5,
                                  use_cuda=False,
                                  double_fit=False, workers=0)
        after_calib_param_init = list(map(lambda x: x.clone(), self.calibrator.init_model.parameters()))
        after_calib_param = list(map(lambda x: x.clone(), self.calibrator.calibrated_model.parameters()))

        assert all([np.allclose(i.detach(), j.detach())
                    for i, j in zip(before_calib_param_init, after_calib_param_init)])

        assert not all([np.allclose(i.detach(), j.detach())
                        for i, j in zip(before_calib_param, after_calib_param)])

    def test_reg_l2_called(self):
        self.calibrator.l2_reg = Mock(return_value=torch.Tensor([0]))
        self.calibrator.calibrate(self.dataset, self.dataset,
                                  batch_size=10, epoch=5,
                                  use_cuda=False,
                                  double_fit=False, workers=0)
        self.calibrator.l2_reg.assert_called()


if __name__ == '__main__':
    pytest.main()

