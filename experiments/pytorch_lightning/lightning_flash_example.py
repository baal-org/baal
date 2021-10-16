from typing import Any

import torch
import argparse
import os

import torch.backends
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus, RunningStage
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import transforms

try:
    import flash
except ImportError as e:
    print(e)
    raise ImportError(
        "`lightning-flash` library is required to run this example."
        " pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[image]'"
    )
from flash.image import ImageClassifier, ImageClassificationData
from flash.core.classification import Probabilities
from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop

from baal.active import get_heuristic

IMG_SIZE = 128

train_transforms = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])


class DataModule_(ImageClassificationData):

    @property
    def num_classes(self):
        return 10


def get_data_module(heuristic, data_path):
    train_set = datasets.CIFAR10(data_path, train=True, download=True)
    test_set = datasets.CIFAR10(data_path, train=False, download=True)
    dm = DataModule_.from_datasets(train_dataset=train_set,
                                   test_dataset=test_set,
                                   train_transform=train_transforms,
                                   test_transform=test_transforms,
                                   batch_size=64, )
    active_dm = ActiveLearningDataModule(dm,
                                         heuristic=get_heuristic(heuristic),
                                         initial_num_labels=1024,
                                         query_size=100,
                                         val_split=0.01)
    assert active_dm.has_test, "No test set?"
    return active_dm


def get_model(dm):
    loss_fn = nn.CrossEntropyLoss()
    head = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, dm.num_classes),
    )
    model = ImageClassifier(num_classes=dm.num_classes,
                            head=head,
                            backbone="vgg16",
                            pretrained=True,
                            loss_fn=loss_fn,
                            optimizer=torch.optim.SGD,
                            optimizer_kwargs={"lr": 0.001,
                                              "momentum": 0.9,
                                              "weight_decay": 0},
                            learning_rate=0.001,
                            # we don't use learning rate here since it is initialized in the optimizer.
                            serializer=Probabilities(), )
    return model


class MyActiveLearningLoop(ActiveLearningLoop):
    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.progress.increment_started()

        if self.trainer.datamodule.has_labelled_data:
            self.fit_loop.run()

        if self.trainer.datamodule.has_test:
            self._reset_testing()
            metrics = self.trainer.test_loop.run()[0]
            self.trainer.logger.log_metrics(metrics, step=self.trainer.global_step)

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_predicting()
            probabilities = self.trainer.predict_loop.run()
            self.trainer.datamodule.label(probabilities=probabilities)
        else:
            raise StopIteration

        self._reset_fitting()
        self.progress.increment_processed()

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        super().on_run_start(*args, **kwargs)
        self.trainer.test_loop._return_predictions = True

    def _reset_testing(self):
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.testing = True
        self.trainer.lightning_module.on_test_dataloader()
        self.trainer.accelerator.connect(self._lightning_module)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        if self.trainer.datamodule.has_labelled_data:
            self._reset_dataloader_for_stage(RunningStage.TRAINING)
            self._reset_dataloader_for_stage(RunningStage.VALIDATING)
            self._reset_dataloader_for_stage(RunningStage.TESTING)
        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_dataloader_for_stage(RunningStage.PREDICTING)
        self.progress.increment_ready()


def main(args):
    gpus = 1 if torch.cuda.is_available() else 0
    active_dm = get_data_module(args.heuristic, args.data_path)
    model = get_model(active_dm.labelled)
    logger = TensorBoardLogger(os.path.join(args.ckpt_path, "tensorboard"),
                               name=f"flash-example-cifar-{args.heuristic}")
    trainer = flash.Trainer(gpus=gpus, max_epochs=2500,
                            default_root_dir=args.ckpt_path, logger=logger, limit_val_batches=0, )
    active_learning_loop = MyActiveLearningLoop(label_epoch_frequency=40,
                                                inference_iteration=20)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop
    trainer.finetune(model, datamodule=active_dm, strategy="freeze")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--data_path", default="/data", type=str)
    parser.add_argument("--ckpt_path", default="/ckpt", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
