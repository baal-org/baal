import structlog
from copy import deepcopy
import torch
from torch.optim import Adam
from torch import nn
from baal.utils.metrics import ECE

log = structlog.get_logger("Calibrating...")


class DirichletCalibrator(object):
    """
    Adding a linear layer to a classifier model after the model is
    trained and train this new layer until convergence.
    Together with the linear layer, the model is now calibrated.
    Source: https://arxiv.org/abs/1910.12656

    Args:

        wrapper (baal.ModelWrapper): Provides training and testing methods.
        num_classes (int): Number of classes in classification task.
        lr (float): Learning rate.
        l (float): Regularization factor for the linear layer weights.
        mu (float): Regularization factor for the linear layer biases.
            If not given, will be initialized by "l".

    """
    def __init__(self, wrapper, num_classes, lr, l, mu=None):
        self.wrapper = wrapper
        self.init_model = deepcopy(wrapper.model)
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.lr =lr
        self.l = l
        self.mu = mu or l

        # TODO: need to support kwargs for initializer for ece_per_class
        self.wrapper.add_metrics("ece", lambda: ECE())
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
        name, params = list(self.model.named_parameters())[-2]
        assert 'weight' in name
        w_l2_factor = params.norm(2)

        name, params = list(self.model.named_parameters())[-1]
        assert 'bias' in name
        b_l2_factor = params.norm(2)

        return self.l * w_l2_factor + self.mu * b_l2_factor


    def calibrate(self, dataset, val_dataset, epoch,
                  batch_size, use_cuda,
                  double_fit=False, **kwargs):
        """
        Training the linear layer given a training set and a validation set.
        The training set should be different from what model is trained on.

        Args:
            dataset (torch.Dataset): The training set.
            val_dataset (torch.Dataset): The validation set.
            epoch (int): Number of epochs to train the linear layer for.
            batch_size (int): Batch size for training and validation.
            use_cuda (bool): If "True", will use GPU.
            double_fit (bool): If "True" would fit twice on the train set.
            kwargs : Rest of parameters for baal.ModelWrapper.train_and_test_on_dataset().

        Returns:
            loss_history (list[float]): List of loss values for each epoch.
            model.state_dict (dict): Model weights.

        """
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        trained_dict = {k: v for k, v in self.wrapper.state_dict().items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(trained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

        # make sure that we are not retraining the model
        for param in self.init_model.parameters():
            param.requires_grad = False

        # reinitialize the dirichlet calibration layer
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))
        if use_cuda:
            self.dirichlet_linear.cuda()


        optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        loss_history, weights = self.wrapper.train_and_test_on_dataset(dataset, val_dataset, optimizer,
                                                                       batch_size, epoch, use_cuda,
                                                                       return_best_weights=True,
                                                                       patience=None, **kwargs)
        self.model.load_state_dict(weights)

        if double_fit:
            self.lr = self.lr / 10
            optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            loss_history, weights = self.wrapper.train_and_test_on_dataset(dataset, val_dataset, optimizer,
                                                                           batch_size, epoch, use_cuda,
                                                                           return_best_weights=True,
                                                                           patience=None, **kwargs)
            self.model.load_state_dict(weights)

        return loss_history, self.model.state_dict()

    def get_calibrated_model(self):
        return self.model

    def get_metrics(self):
        return self.wrapper.metrics


