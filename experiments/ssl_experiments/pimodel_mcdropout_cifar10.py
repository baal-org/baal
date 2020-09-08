import argparse
import copy
from argparse import Namespace

import torch
from baal.active import ActiveLearningDataset
from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module
from baal.utils.pytorch_lightning import ActiveLearningMixin, BaalTrainer, ResetCallback
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16

from experiments.ssl_experiments.pimodel_cifar10 import PIModel


class PIActiveLearningModel(ActiveLearningMixin, PIModel):
    def __init__(self, active_dataset: ActiveLearningDataset, hparams: Namespace, network: nn.Module):
        super().__init__(active_dataset, hparams, network)

        self.network = patch_module(self.network)

    def pool_loader(self):
        return DataLoader(self.active_dataset.pool, self.hparams.batch_size, shuffle=False)

    def epoch_end(self, outputs):
        out = super().epoch_end(outputs)
        out['log']['active_set_len'] = len(self.actipoove_dataset)

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
        parser = super(PIActiveLearningModel, PIActiveLearningModel).add_model_specific_args(parent_parser)
        parser.add_argument('--query_size', type=int, default=100)
        parser.add_argument('--max_sample', type=int, default=-1)
        parser.add_argument('--iterations', type=int, default=20)
        parser.add_argument('--replicate_in_memory', action='store_true')
        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument('--data-root', default='/tmp', type=str, help='Where to download the data')
    args.add_argument('--gpus', default=torch.cuda.device_count(), type=int)
    args = PIActiveLearningModel.add_model_specific_args(args)
    params = args.parse_args()

    print(params)

    active_set = ActiveLearningDataset(
        CIFAR10(params.data_root, train=True, transform=PIModel.train_transform, download=True),
        pool_specifics={'transform': PIModel.test_transform},
        make_unlabelled=lambda x: x[0])
    active_set.label_randomly(100)

    print("Active set length: {}".format(len(active_set)))
    print("Pool set length: {}".format(len(active_set.pool)))

    heuristic = BALD()
    net = vgg16(pretrained=False, num_classes=10)
    model = PIActiveLearningModel(network=net, active_dataset=active_set, hparams=params)

    dp = 'dp' if params.gpus > 1 else None
    trainer = BaalTrainer(max_epochs=params.epochs, default_root_dir=params.data_root,
                          gpus=params.n_gpus, distributed_backend=dp,
                          # The weights of the model will change as it gets
                          # trained; we need to keep a copy (deepcopy) so that
                          # we can reset them.
                          callbacks=[ResetCallback(copy.deepcopy(model.state_dict()))],
                          dataset=active_set,
                          heuristic=heuristic,
                          ndata_to_label=params.query_size
                          )

    AL_STEPS = 100
    for al_step in range(AL_STEPS):
        print(f'Step {al_step} Dataset size {len(active_set)}')
        trainer.fit(model)
        should_continue = trainer.step()
        if not should_continue:
            break
