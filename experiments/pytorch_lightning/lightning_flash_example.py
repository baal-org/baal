import torch
import argparse
import os

import torch.backends
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
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
                                   val_transform=test_transforms,
                                   test_transform=test_transforms)
    active_dm = ActiveLearningDataModule(dm,
                                         heuristic=get_heuristic(heuristic),
                                         initial_num_labels=1024,
                                         query_size=100,
                                         val_split=0.1)
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


def main(args):
    gpus = 1 if torch.cuda.is_available() else 0
    active_dm = get_data_module(args.heuristic, args.data_path)
    model = get_model(active_dm.labelled)
    logger = TensorBoardLogger(os.path.join(args.ckpt_path, "tensorboard"), name="flash-example-cifar")
    trainer = flash.Trainer(gpus=gpus, max_epochs=2500,
                            default_root_dir=args.ckpt_path, logger=logger)
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=40,
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
