import argparse
import os
from functools import partial

import structlog
import torch
import torch.backends
from flash.core.classification import LogitsOutput
from pytorch_lightning import seed_everything
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
        " pip install 'git+https://github.com/PyTorchLightning/"
        "lightning-flash.git#egg=lightning-flash[image]'"
    )
from flash.image import ImageClassifier, ImageClassificationData
from flash.image.classification.integrations.baal import (
    ActiveLearningDataModule,
    ActiveLearningLoop,
)

from baal.active import get_heuristic

log = structlog.get_logger()

IMG_SIZE = 128

train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class DataModule_(ImageClassificationData):
    @property
    def num_classes(self):
        return 10


def get_data_module(heuristic, data_path):
    train_set = datasets.CIFAR10(data_path, train=True, download=True)
    test_set = datasets.CIFAR10(data_path, train=False, download=True)
    dm = DataModule_.from_datasets(
        train_dataset=train_set,
        test_dataset=test_set,
        train_transform=train_transforms,
        test_transform=test_transforms,
        # Do not forget to set `predict_transform`,
        # this is what we will use for uncertainty estimation!
        predict_transform=test_transforms,
        batch_size=64,
    )
    active_dm = ActiveLearningDataModule(
        dm,
        heuristic=get_heuristic(heuristic),
        initial_num_labels=1024,
        query_size=100,
        val_split=0.0,
    )
    assert active_dm.has_test, "No test set?"
    return active_dm


def get_model(dm):
    loss_fn = nn.CrossEntropyLoss()
    head = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, dm.num_classes),
    )
    LR = 0.001
    model = ImageClassifier(
        num_classes=dm.num_classes,
        head=head,
        backbone="vgg16",
        pretrained=True,
        loss_fn=loss_fn,
        optimizer=partial(torch.optim.SGD, momentum=0.9, weight_decay=5e-4),
        learning_rate=LR,
    )
    model.output = (
        LogitsOutput()
    )  # Note the serializer to Logits to be able to estimate uncertainty.

    return model


def main(args):
    seed_everything(args.seed)
    gpus = 1 if torch.cuda.is_available() else 0
    active_dm: ActiveLearningDataModule = get_data_module(args.heuristic, args.data_path)
    model: ImageClassifier = get_model(active_dm.labelled)
    logger = TensorBoardLogger(
        os.path.join(args.ckpt_path, "tensorboard"),
        name=f"flash-example-cifar-{args.heuristic}-{args.seed}",
    )
    # We use Flash trainer without validation set.
    # In practice, using a validation set is risky because we overfit often.
    trainer = flash.Trainer(
        gpus=gpus,
        max_epochs=2500,
        default_root_dir=args.ckpt_path,
        logger=logger,
        limit_val_batches=0,
    )

    # We will train for 20 epochs before doing 20 MC-Dropout iterations to estimate uncertainty.
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=20, inference_iteration=20)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop
    # We do not freeze the backbone, this gives better performance.
    trainer.finetune(model, datamodule=active_dm, strategy="no_freeze")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heuristic",
        default="bald",
        type=str,
        choices=["bald", "random", "entropy"],
        help="Which heuristic to select.",
    )
    parser.add_argument("--data_path", default="/data", type=str, help="Where to find the dataset.")
    parser.add_argument("--ckpt_path", default="/ckpt", type=str, help="Where to save checkpoints")
    parser.add_argument("--seed", default=2021, type=int, help="Random seed of the experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
