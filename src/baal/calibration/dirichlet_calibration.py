import structlog
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from baal.utils.cuda_utils import to_cuda
from baal.utils.metrics import Loss

log = structlog.get_logger("Dirichlet Calibrator")

class DirichletCalibrator:
    """
    Adding a linear layer to a classifier model after the model is
    trained and train this new layer until convergence.
    Together with the linear layer, the model is now calibrated.
    Source: https://arxiv.org/abs/1910.12656

    Args:
        model (nn.Module): Pytorch model.
        weights (torch.Tensor): Model weights after training.
        num_classes (int): Number of classes in classification task.
        lr (float): Learning rate.
        l (float): Regularization factor for the linear layer weights.
        mu (float): Regularization factor for the linear layer biases.
            If not given, will be initialized by "l".

    """
    def __init__(self, model, weights, num_classes, lr, l, mu=None):
        self.init_model = model
        self.trained_weights = weights
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.lr =lr
        self.l = l
        self.mu = mu or l
        self.avg_training_loss = Loss()
        self.avg_test_loss = Loss()
        self.drichletlinear = nn.Linear(self.num_classes, self.num_classes)
        self.model = nn.Sequential(
            self.init_model,
            self.drichletlinear
        )

    def load_state_dict(self, weights):
        """ we should start from the weights of a trained model.
        """
        return self.model.load_state_dict(weights)

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


    def calibrate(self, trained_dict, dataset, val_dataset, epoch, batch_size,
                  double_fit=False, patience=5, workers=4, use_cuda=True):
        """
        Training the linear layer given a training set and a validation set.
        The training set should be different from what model is trained on.

        Args:
            trained_dict (dict): Pytorch state_dict of the trained model.
            dataset (torch.Dataset): The training set.
            val_dataset (torch.Dataset): The validation set.
            epoch (int): Number of epochs to train the linear layer for.
            batch_size (int): Batch size for training and validation.
            double_fit (bool): If "True" would fit twice on the train set.
            patience (int): Number of epochs to check the loss plateau
                for early stopping.
            workers (int): Number of workers to use for training and evaluation.
            use_cuda (bool): If "True" will train on GPU.

        Returns:
            loss_history (list[float]): List of loss values for each epoch.
            model.state_dict (dict): Model weights.

        """
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(trained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

        # make sure that we are not retraining the model
        for param in self.init_model.parameters():
            param.requires_grad = False

        # reinitialize the dirichlet calibration layer
        self.drichletlinear.weight.data.copy_(torch.eye(self.drichletlinear.weight.shape[0]))
        self.drichletlinear.bias.data.copy_(torch.zeros(*self.drichletlinear.bias.shape))
        if use_cuda:
            self.drichletlinear.cuda()


        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        loss_history, weights = self.train(dataset, val_dataset, epoch, patience, batch_size, workers, use_cuda)
        self.model.load_state_dict(weights)

        if double_fit:
            self.lr = self.lr / 10
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            loss_history, weights = self.train(dataset, val_dataset, epoch, patience, batch_size, workers, use_cuda)
            self.model.load_state_dict(weights)

        return loss_history, self.model.state_dict()

    def train(self, dataset, val_dataset, epoch, patience, batch_size, workers, use_cuda)
        """
         Trainer for the linear layer using self.l2_reg().
        Args:
            dataset (torch.Dataset): The training set.
            val_dataset (torch.Dataset): The validation set.
            epoch (int): Number of epochs to train the linear layer for.
            patience (int): Number of epochs to check the loss plateau
                for early stopping.
            batch_size (int): Batch size for training and validation.
            workers (int): Number of workers to use for training and evaluation.
            use_cuda (bool): If "True" will train on GPU.

        Returns:
            history (List[float]): List of loss values for each epoch.
            best_weight (dict): Best model weights.

        """
        best_weights = None
        best_loss = 1e10
        best_epoch = 0
        history = []
        for e in range(epoch):
            self.model.train()
            log.info("Starting training", epoch=epoch, dataset=len(dataset))
            self.avg_training_loss.reset()
            for data, target in DataLoader(dataset, batch_size, shuffle=True, num_workers=workers):
                if use_cuda:
                    data, target = to_cuda(data), to_cuda(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss = loss + self.l2_reg()
                loss.backward()
                self.optimizer.step()
                self.avg_training_loss.update(loss)
            history.append({'avg_train_loss': self.avg_training_loss.value})
            self.optimizer.zero_grad()  # Assert that the gradient is flushed.
            log.info("Training complete", train_loss=self.avg_training_loss.value)

            self.model.eval()
            log.info("Starting evaluating", dataset=len(val_dataset))
            self.avg_test_loss.reset()
            for data, target in DataLoader(val_dataset, batch_size,
                                           shuffle=False, num_workers=workers):
                with torch.no_grad():
                    if use_cuda:
                        data, target = to_cuda(data), to_cuda(target)
                    preds = self.model(data)
                    te_loss = self.loss(preds, target)
                    self.avg_test_loss.update(te_loss)
            history.append({'avg_test_loss': self.avg_test_loss.value})
            log.info("Evaluation complete", test_loss=self.avg_test_loss.value)

            if te_loss < best_loss:
                best_epoch = e
                best_loss = te_loss
                best_weights = deepcopy(self.model.state_dict())

            if patience is not None and (e - best_epoch) > patience:
                # Early stopping
                break

        return history, best_weights


