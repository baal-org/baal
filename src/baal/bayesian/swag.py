import itertools
from typing import List, Tuple, Iterable
from collections import defaultdict
import torch
from torch.optim import SGD
from torch.optim.optimizer import required


class StochasticWeightAveraging(torch.optim.Optimizer):
    """Optimise a model using stochastic weight averaging.

    This class optimises using standard SGD intially, but when you call
    optimiser.swa(), your model parameters are converted to the average of
    some recent points along the gradient descent path.

    You can call optimiser.sgd() to reset the parameters to the most recent
    gradient descent point.

    Parameters
    ----------
    base_optimizer (torch.optim.Optimizer):
        A pytorch optimizer. StochasticWeightAveraging uses this optimizer to
        perform optimization, and will sample at various points along the way.
    swa_start (int):
        The number of optimization steps after which stochastic weight averaging
        should start. This is the number of *batches* to process, not the number
        of epochs.
    swa_freq (int):
        The number of steps (i.e. batches) in between stochastic weight averaging
        samples. This is often one epoch.
    collect_covariance (bool, default True):
        Whether to collect multiple individual samples in addition to the mean
        and variance in order to calculate a low-rank covariance matrix.
    n_deviations : int
        How many deviations from the mean to store. These will be used to
        estimate the covariance between weights.
    storage_device : torch.device
        Where to store the Stochastic Weight samples. If you have space on the
        GPU, use that - otherwise, use the CPU (the default).
    cycle_learning_rate : bool
        Whether to cycle the learning rate in between multiple values. Raising
        the learning rate and gradually reducing it is helpful for stochastic
        weight averaging as it allows traversing of a wider range of the
        parameter space.
    lr_min : float
        The low end of the learning rate cycle schedule.
    lr_max : float
        The high end of the learning rate cycle schedule.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        swa_start: int = required,
        swa_freq: int = required,
        collect_covariance: bool = True,
        n_deviations: int = 20,
        storage_device: torch.device = torch.device("cpu"),
        cycle_learning_rate: bool = True,
        lr_min: float = required,
        lr_max: float = required,
    ):
        self.state = defaultdict(dict)
        self.base_optimizer = base_optimizer
        self.storage_device = storage_device
        self.collect_covariance = collect_covariance
        if swa_freq is required:
            raise ValueError(
                "You need to pass a number of steps after which averaging "
                "occurs, e.g. swa_steps=epoch_size // batch_size."
            )
        elif swa_freq == 0:
            raise ValueError("swa_freq needs to be a positive integer.")
        self.swa_freq = swa_freq
        if swa_start is required:
            raise ValueError(
                "You need to pass a number of steps for which no averaging "
                "happens, e.g. swa_burn_in=100 * epoch_size // batch_size."
            )
        self.swa_start = swa_start

        if cycle_learning_rate and (lr_min is required or lr_max is required):
            raise ValueError(
                "If cycle_learning_rate is True, you need to provide "
                "a min and max for the learning rate to be cycled between."
            )
        elif cycle_learning_rate:
            self.lr_min = lr_min
            self.lr_max = lr_max
        else:
            self.lr_min = None
            self.lr_max = None

        self.state['n_steps'] = 0
        self.state['samples_taken'] = 0
        self.state['n_deviations'] = max((n_deviations, 1))
        self.distribution_cache = {}

    def step(self, *args, **kwargs):
        # first, execute a step with base optimizer
        self.base_optimizer.step(*args, **kwargs)
        self.state['n_steps'] += 1

        # check if n_steps indicates we need to update averages
        if self.state['n_steps'] >= self.swa_start and self.progress_between_samples == 0:
            self._update_means()

        # set the learning rate for the next step:
        if self.lr_max is not None and self.state['n_steps'] >= self.swa_start:
            self._set_lr(self.lr_max - self.progress_between_samples * (self.lr_max - self.lr_min))

    @property
    def progress_between_samples(self):
        return ((self.state['n_steps'] - self.swa_start) / self.swa_freq) % 1

    @property
    def samples_taken(self):
        return self.state['samples_taken']

    @samples_taken.setter
    def samples_taken(self, value):
        self.state['samples_taken'] = value

    def _set_lr(self, value):
        for group in self.param_groups:
            group["lr"] = value

    def _update_means(self):

        for group in self.param_groups:
            for p in group["params"]:
                p_state = self.state[p]
                values = p.data.detach().to(self.storage_device, non_blocking=True)

                if "mean" not in p_state:
                    p_state.setdefault("mean", torch.zeros_like(values))

                delta_mean = (values - p_state["mean"]) / (self.samples_taken + 1)
                p_state["mean"] += delta_mean

                # use welford's algorithm to keep track of variance:
                if "welfords_square_distance" not in p_state:
                    p_state.setdefault("welfords_square_distance", torch.zeros_like(values))

                p_state["welfords_square_distance"] += (values - (p_state["mean"] - delta_mean)) * (
                    values - p_state["mean"]
                )

                # update the deviation matrix
                if "deviations" not in p_state:
                    p_state["deviations"] = torch.zeros_like(values).unsqueeze(-1)
                else:
                    p_state["deviations"] = torch.cat(
                        (p_state["deviations"], (values - p_state["mean"]).unsqueeze(-1)), dim=-1
                    )
                # trim the matrix if needed
                if p_state["deviations"].size(-1) > self.state["n_deviations"]:
                    p_state["deviations"] = p_state["deviations"][..., 1:]

        self.samples_taken += 1

    def swa(self):
        """Apply the stochastic weight average to the model."""
        if (self.state['n_steps'] - self.swa_start) // self.swa_freq > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data.copy_(self.state[p]["mean"], non_blocking=True)

    def sgd(self):
        """
        Un-apply the stochastic weight averaging to the model.

        Note that this only resets to the last point at which you did a SWA
        update, which is not necessarily the most recent stochastic gradient
        descent point.
        """
        if (self.state['n_steps'] - self.swa_start) // self.swa_freq > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data.copy_(
                        self.state[p]["mean"] + self.state[p]["deviations"][..., -1],
                        non_blocking=True,
                    )

    def bn_update(self, model, dataloader, device=None):
        """
        Update the batch norm statistics in a model.

        Parameters
        ----------
        model : torch.nn.Module
            The model.
        dataloader : torch.utils.data.DataLoader
            The data to use for computing the batch-norm stats.
        device : torch.device, optional
            Where to move the data before passing it to the model.
        """
        if device is None:
            device = next(model.parameters()).device
        if not any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
        ):
            return

        was_training = model.training
        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()

        with torch.no_grad():
            for input_ in dataloader:
                if isinstance(input_, (list, tuple)):
                    input_ = input_[0]

                input_ = input_.to(device)
                model(input_)

        model.train(was_training)

    def sample(self, scale=1.0, blockwise=True):
        """
        Sample parameters and apply to the model being optimised.

        NB: Currently this only draws a diagonal sample and does not implement
        covariance between weights.

        Parameters
        ----------
        scale : float
            The amount by which to scale the variance of the distribution from
            which to draw samples. If zero, only the mean gets drawn. If more
            than 1.0, the distribution is broadened.
        blockwise : bool
            If sampling with covariance, whether to consider covariance between
            each parameter block (blockwise=True, default) or for the entire
            network (blockwise=False), which can take up a lot of memory.
        """
        if not (self.state['n_steps'] - self.swa_start) // self.swa_freq > 0:
            raise ValueError("You haven't made any SWA updates yet.")

        if not blockwise:
            # iterate over all parameters:
            for param, sample in zip(
                self._all_params(),
                self._draw_sample(self._all_params(), scale, self.collect_covariance),
            ):
                param.data.copy_(sample, non_blocking=True)

        else:
            # iterate over the network in chunks:
            for param_group in grouper(self._all_params(), 1):
                for param, sample in zip(
                    param_group, self._draw_sample(param_group, scale, self.collect_covariance)
                ):
                    param.data.copy_(sample, non_blocking=True)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def _all_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                yield p

    def _draw_sample(
        self, parameters: Iterable[torch.nn.Parameter], scale: float, covariance: bool
    ):
        means = []
        variances = []
        deviations = []
        for p in parameters:
            means.append(self.state[p]["mean"])
            variances.append(self.state[p]["welfords_square_distance"] / self.samples_taken)
            if covariance:
                deviations.append(self.state[p]["deviations"])

        means = torch.cat([mean.flatten() for mean in means])
        variances = torch.cat([var.clamp(1e-10).flatten() for var in variances])

        if covariance:
            deviations = torch.cat([deviation.flatten(0, -2) for deviation in deviations])

        if covariance:
            diag_sample = torch.randn_like(variances) * variances.sqrt()
            cov_sample = deviations @ torch.randn(deviations.size(1))
            cov_sample /= (deviations.size(-1) - 1) ** 0.5
            sample = means + 0.5 * (diag_sample + cov_sample) * scale ** 0.5
        else:
            diag_sample = torch.randn_like(variances) * variances.sqrt()
            sample = means + diag_sample * scale ** 0.5

        return _unflatten_like(sample, parameters)

    def get_parameter_distribution(
        self, parameters: Iterable[torch.nn.Parameter], scale: float, covariance: bool
    ) -> torch.distributions.Distribution:  # nocov

        means = []
        square_means = []
        deviations = []
        for p in parameters:
            means.append(self.state[p]["mean"])
            square_means.append(self.state[p]["square_mean"])
            if covariance:
                deviations.append(self.state[p]["deviations"])

        means = torch.cat([mean.flatten() for mean in means])
        square_means = torch.cat([square_mean.flatten() for square_mean in square_means])
        variances = scale * (square_means - means ** 2).clamp(1e-20)

        if not covariance:
            dist = torch.distributions.Normal(means, variances.sqrt())
        else:
            deviations = torch.cat([deviation.flatten(0, -2) for deviation in deviations])
            dist = torch.distributions.LowRankMultivariateNormal(
                means, deviations / (deviations.size(-1) - 1), variances
            )

        return dist


def _unflatten_like(params: torch.Tensor, param_list: List[torch.Tensor]) -> List[torch.Tensor]:
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    # shaped like likeTensorList
    output_list = []
    i = 0
    for tensor in param_list:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        output_list.append(params[i : (i + n)].view(tensor.shape))
        i += n
    return output_list


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
