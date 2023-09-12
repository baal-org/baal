import argparse
import copy
from argparse import Namespace

import torch
from experiments.ssl_experiments.pimodel_cifar10 import PIModel
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16

from baal import ActiveLearningDataset, get_heuristic
from baal.bayesian.dropout import patch_module
from baal import ActiveLightningModule, BaalTrainer, ResetCallback


class PIActiveLearningModel(ActiveLightningModule, PIModel):
    def __init__(
        self, active_dataset: ActiveLearningDataset, hparams: Namespace, network: nn.Module
    ):
        super().__init__(active_dataset, hparams, network)

        self.network = patch_module(self.network)

    def pool_dataloader(self):
        return DataLoader(
            self.active_dataset.pool, self.hparams.batch_size, shuffle=False, num_workers=4
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure, **kwargs):
        optimizer.step()
        optimizer.zero_grad()

    def test_epoch_end(self, outputs):
        out = super().test_epoch_end(outputs)
        out["log"]["active_set_len"] = len(self.active_dataset)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific arguments to argparser.

        Args:
            parent_parser (argparse.ArgumentParser): parent parser to which to add arguments

        Returns:
            argparser with added arguments
        """
        parser = super(PIActiveLearningModel, PIActiveLearningModel).add_model_specific_args(
            parent_parser
        )
        parser.add_argument("--query_size", type=int, default=100)
        parser.add_argument("--max_sample", type=int, default=-1)
        parser.add_argument("--iterations", type=int, default=20)
        parser.add_argument("--heuristic", type=str, default="bald")
        parser.add_argument("--replicate_in_memory", action="store_true")
        return parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument("--data-root", default="/tmp", type=str, help="Where to download the data")
    args.add_argument("--gpus", default=torch.cuda.device_count(), type=int)
    args = PIActiveLearningModel.add_model_specific_args(args)
    params = args.parse_args()

    active_set = ActiveLearningDataset(
        CIFAR10(params.data_root, train=True, transform=PIModel.train_transform, download=True),
        pool_specifics={"transform": PIModel.test_transform},
    )
    active_set.label_randomly(500)

    print("Active set length: {}".format(len(active_set)))
    print("Pool set length: {}".format(len(active_set.pool)))

    heuristic = get_heuristic(params.heuristic)
    model = vgg16(weights=None, num_classes=10)
    weights = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth")
    weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
    model.load_state_dict(weights, strict=False)
    model = PIActiveLearningModel(network=model, active_dataset=active_set, hparams=params)

    dp = "dp" if params.gpus > 1 else None
    trainer = BaalTrainer(
        max_epochs=params.epochs,
        default_root_dir=params.data_root,
        gpus=params.gpus,
        distributed_backend=dp,
        # The weights of the model will change as it gets
        # trained; we need to keep a copy (deepcopy) so that
        # we can reset them.
        callbacks=[ResetCallback(copy.deepcopy(model.state_dict()))],
        dataset=active_set,
        heuristic=heuristic,
        query_size=params.query_size,
    )

    AL_STEPS = 2000
    for al_step in range(AL_STEPS):
        # TODO fix this
        trainer.current_epoch = 0
        print(f"Step {al_step} Dataset size {len(active_set)}")
        trainer.fit(model)
        should_continue = trainer.step()
        if not should_continue:
            break
