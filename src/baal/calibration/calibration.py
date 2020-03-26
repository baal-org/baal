from copy import deepcopy

import structlog
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset

from baal import ModelWrapper
from baal.utils.metrics import ECE, ECE_PerCLs

log = structlog.get_logger("Calibrating...")


class DirichletCalibrator(object):
    """
    Adding a linear layer to a classifier model after the model is
    trained and train this new layer until convergence.
    Together with the linear layer, the model is now calibrated.
    Source: https://arxiv.org/abs/1910.12656
    Code inspired from: https://github.com/dirichletcal/experiments_neurips

    References:
        @article{kullbeyond,
                title={Beyond temperature scaling: Obtaining well-calibrated multi-class
                 probabilities with Dirichlet calibration Supplementary material},
                author={Kull, Meelis and Perello-Nieto,
                 Miquel and K{\"a}ngsepp, Markus and Silva Filho,
                  Telmo and Song, Hao and Flach, Peter}
                }

    Args:

        wrapper (ModelWrapper): Provides training and testing methods.
        num_classes (int): Number of classes in classification task.
        lr (float): Learning rate.
        reg_factor (float): Regularization factor for the linear layer weights.
        mu (float): Regularization factor for the linear layer biases.
            If not given, will be initialized by "l".

    """

    def __init__(self, wrapper: ModelWrapper, num_classes: int, lr: float,
                 reg_factor: float, mu: float = None):
        self.wrapper = wrapper
        self.init_model = deepcopy(wrapper.model)
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.reg_factor = reg_factor
        self.mu = mu or reg_factor

        self.wrapper.add_metric("ece", lambda: ECE())
        self.wrapper.add_metric("ece", lambda: ECE_PerCLs(num_classes))
        self.dirichlet_linear = nn.Linear(self.num_classes, self.num_classes)
        self.model = nn.Sequential(
            self.init_model,
            self.dirichlet_linear
        )

    def l2_reg(self):
        """ Using trainable layer's parameters for l2 regularization.

        Returns:
            The regularization term for the linear layer.
        """
        weight_p, bias_p = self.dirichlet_linear.parameters()
        w_l2_factor = weight_p.norm(2)
        b_l2_factor = bias_p.norm(2)
        return self.reg_factor * w_l2_factor + self.mu * b_l2_factor

    def calibrate(self, train_set: Dataset, test_set: Dataset,
                  batch_size: int, epoch: int, use_cuda: bool,
                  double_fit: bool = False, **kwargs):
        """
        Training the linear layer given a training set and a validation set.
        The training set should be different from what model is trained on.

        Args:
            train_set (Dataset): The training set.
            test_set (Dataset): The validation set.
            batch_size (int): Batch size used.
            epoch (int): Number of epochs to train the linear layer for.
            use_cuda (bool): If "True", will use GPU.
            double_fit (bool): If "True" would fit twice on the train set.
            kwargs (dict): Rest of parameters for baal.ModelWrapper.train_and_test_on_dataset().

        Returns:
            loss_history (list[float]): List of loss values for each epoch.
            model.state_dict (dict): Model weights.

        """
        model_dict = self.init_model.state_dict()

        # 1. filter out unnecessary keys
        trained_dict = {k: v for k, v in self.wrapper.state_dict().items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(trained_dict)
        # 3. load the new state dict
        self.init_model.load_state_dict(model_dict)

        # reinitialize the dirichlet calibration layer
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))
        if use_cuda:
            self.dirichlet_linear.cuda()

        optimizer = Adam(self.dirichlet_linear.parameters(), lr=self.lr)

        # making sure that the training is done on linear layer
        self.wrapper.model = self.model

        loss_history, weights = self.wrapper.train_and_test_on_datasets(train_set, test_set,
                                                                        optimizer, batch_size,
                                                                        epoch, use_cuda,
                                                                        return_best_weights=True,
                                                                        patience=None, **kwargs)
        self.model.load_state_dict(weights)

        if double_fit:
            self.lr = self.lr / 10
            optimizer = Adam(self.dirichlet_linear.parameters(), lr=self.lr)
            loss_history, weights = self.wrapper.train_and_test_on_datasets(
                train_set, test_set,
                optimizer, batch_size,
                epoch, use_cuda,
                return_best_weights=True,
                patience=None,
                **kwargs)
            self.lr = self.lr * 10
            self.model.load_state_dict(weights)

        self.wrapper.model = self.init_model

        return loss_history, self.model.state_dict()

    @property
    def calibrated_model(self):
        return self.model

    @property
    def metrics(self):
        return self.wrapper.metrics
