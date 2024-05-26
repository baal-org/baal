import os

import numpy as np
import pytest
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torchvision.models import vgg
from torchvision.transforms import ToTensor, Resize, Compose

from baal.active import ActiveLearningDataset
from baal.active import ActiveLearningLoop
from baal.active import heuristics
from baal.modelwrapper import ModelWrapper, TrainingArgs
from baal.calibration import DirichletCalibrator


class DummyDataset(Dataset):
    def __init__(self, t=None):
        self.transform = t

    def __len__(self):
        return 50

    def __getitem__(self, item):
        i = Image.fromarray(np.random.randint(0, 255, [30, 30, 3], dtype=np.uint8))
        if self.transform:
            i = self.transform(i)
        return (i, item % 10)


@pytest.mark.skipif('CIRCLECI' in os.environ, reason="Really slow")
def test_integration():
    transform_pipeline = Compose([Resize((64, 64)), ToTensor()])
    cifar10_train = DummyDataset(transform_pipeline)
    cifar10_test = DummyDataset(transform_pipeline)

    al_dataset = ActiveLearningDataset(cifar10_train, pool_specifics={'transform': transform_pipeline})
    al_dataset.label_randomly(10)

    use_cuda = False
    model = vgg.vgg11(pretrained=False,
                      num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # We can now use BaaL to create the active learning loop.
    args = TrainingArgs(criterion=criterion, optimizer=optimizer, batch_size=10, use_cuda=use_cuda, epoch=1)
    model = ModelWrapper(model, args)
    # We create an ActiveLearningLoop that will automatically label the most uncertain samples.
    # In this case, we use the widely used BALD heuristic.

    active_loop = ActiveLearningLoop(al_dataset,
                                     model.predict_on_dataset,
                                     heuristic=heuristics.BALD(),
                                     iterations=2,
                                     query_size=10)

    # We're all set!
    num_steps = 10
    for step in range(num_steps):
        old_param = list(map(lambda x: x.clone(), model.model.parameters()))
        model.train_on_dataset(al_dataset)
        model.test_on_dataset(cifar10_test)

        if not active_loop.step():
            break
        new_param = list(map(lambda x: x.clone(), model.model.parameters()))
        assert any([not np.allclose(i.detach(), j.detach())
                    for i, j in zip(old_param, new_param)])
    assert step == 4  # 10 + (4 * 10) = 50, so it stops at iterations 4


@pytest.mark.skipif('CIRCLECI' in os.environ, reason="slow")
def test_calibration_integration():
    transform_pipeline = Compose([Resize((64, 64)), ToTensor()])
    cifar10_train = DummyDataset(transform_pipeline)
    cifar10_test = DummyDataset(transform_pipeline)

    # we don't create different trainset for calibration since the goal is not
    # to calibrate
    al_dataset = ActiveLearningDataset(cifar10_train, pool_specifics={'transform': transform_pipeline})
    al_dataset.label_randomly(10)
    use_cuda = False
    model = vgg.vgg16(pretrained=False,
                      num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    args = TrainingArgs(criterion=criterion, optimizer=optimizer, batch_size=10, use_cuda=use_cuda, epoch=1)
    wrapper = ModelWrapper(model, args)
    calibrator = DirichletCalibrator(wrapper=wrapper, num_classes=10,
                                     lr=0.001, reg_factor=0.01)


    for step in range(2):
        wrapper.train_on_dataset(al_dataset)

        wrapper.test_on_dataset(cifar10_test)


        before_calib_param = list(map(lambda x: x.clone(), wrapper.model.parameters()))

        calibrator.calibrate(al_dataset, cifar10_test, use_cuda=use_cuda, double_fit=False)

        after_calib_param = list(map(lambda x: x.clone(), model.parameters()))


        assert all([np.allclose(i.detach(), j.detach())
                    for i, j in zip(before_calib_param, after_calib_param)])

        assert len(list(wrapper.model.modules())) < len(list(calibrator.calibrated_model.modules()))


if __name__ == '__main__':
    pytest.main()
