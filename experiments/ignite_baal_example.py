import numpy as np
import torch
from ignite.engine import Engine, Events, create_supervised_trainer
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import ActiveLearningDataset
from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import mc_inference

"""Notes:
Things lacking:
    * Reset weights on train begin to initial weights.
    * Log to MLFLow
    * On reload, load active_dataset state
    * On save, save active_dataset state
    * Support for "max_pool_size"
"""


def create_mc_inference_predictor(model, iterations, replicate_in_memory, use_cuda):
    def _inference(engine, batch):
        with torch.no_grad():
            data = batch[0]
            if use_cuda:
                data = data.cuda()
            pred = mc_inference(model, data, iterations, replicate_in_memory)
            engine.state.predictions.append(pred)
            return pred

    engine = Engine(_inference)
    engine.add_event_handler(Events.STARTED, lambda eng: model.eval())

    def _init_preds(engi):
        engi.state.predictions = []

    engine.add_event_handler(Events.STARTED, _init_preds)

    return engine


class DataConfig:
    def __init__(self, active_dataset, hparams):
        super().__init__()
        self.active_dataset = active_dataset
        self.hparams = hparams
        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])

    def train_dataloader(self):
        return DataLoader(self.active_dataset, self.hparams.batch_size, shuffle=True,
                          num_workers=4)

    def test_dataloader(self):
        ds = CIFAR10(root=self.hparams.data_root, train=False,
                     transform=self.test_transform, download=True)
        return DataLoader(ds, self.hparams.batch_size, shuffle=False,
                          num_workers=4)

    def pool_loader(self):
        return DataLoader(self.active_dataset.pool, self.hparams.batch_size, shuffle=False,
                          num_workers=4)


class HParams(BaseModel):
    batch_size: int = 10
    data_root: str = '/tmp'
    num_classes: int = 10
    learning_rate: float = 0.001
    query_size: int = 100
    max_sample: int = -1
    iterations: int = 20
    replicate_in_memory: bool = True


def main(hparams):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
    active_set = ActiveLearningDataset(
        CIFAR10(hparams.data_root, train=True, transform=train_transform, download=True),
        pool_specifics={
            'transform': test_transform
        })
    active_set.label_randomly(100)
    data_cfg = DataConfig(active_set, hparams)
    heuristic = BALD()
    model = vgg16(num_classes=10)
    model = patch_module(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(model, optimizer, loss)
    predictor = create_mc_inference_predictor(model,
                                              iterations=hparams.iterations,
                                              replicate_in_memory=True,
                                              use_cuda=False)

    train_loader = data_cfg.train_dataloader()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.COMPLETED)
    def active_learning(trainer):
        pool_loader = data_cfg.pool_loader()
        predictor.run(tqdm(pool_loader), max_epochs=1, epoch_length=10)
        output = np.vstack(predictor.state.predictions)
        if len(output) > 0:
            to_label = heuristic(output)
            active_set.label(to_label[:hparams.query_size])

    AL_STEPS = 100
    for al_step in range(AL_STEPS):
        trainer.run(train_loader, max_epochs=1)
        print(f"Num. label: {len(active_set)}")

    assert len(active_set) > 100


if __name__ == '__main__':
    main(HParams())
