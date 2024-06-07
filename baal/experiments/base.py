import itertools
from typing import Union, Optional, Any

import numpy as np
import structlog
from torch.utils.data import Subset
from tqdm.autonotebook import tqdm

from baal import ModelWrapper, ActiveLearningDataset
from baal.active.dataset.base import Dataset
from baal.active.heuristics import AbstractHeuristic
from baal.active.stopping_criteria import StoppingCriterion, LabellingBudgetStoppingCriterion
from baal.experiments import FrameworkAdapter
from baal.experiments.modelwrapper import ModelWrapperAdapter

try:
    import transformers
    from baal.transformers_trainer_wrapper import BaalTransformersTrainer
    from baal.experiments.transformers import TransformersAdapter

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    BaalTransformersTrainer = None  # type: ignore
    TransformersAdapter = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

log = structlog.get_logger("baal")


class ActiveLearningExperiment:
    """Experiment manager for Baal.

    Takes care of:
        1. Train the model on the label set.
        2. Evaluate the model on the evaluation set.
        3. Predict on the unlabelled examples.
        4. Label the most uncertain examples.
        5. Stop the experiment if finished.

    Args:
        trainer: Huggingface or ModelWrapper to train
        al_dataset: Active learning dataset
        eval_dataset: Evaluation Dataset
        heuristic: Uncertainty heuristic to use
        query_size: How many items to label per round.
        iterations: How many MC sampling to perform per item.
        pool_size: Optionally limit the size of the unlabelled pool.
        criterion: Stopping criterion for the experiment.
    """

    def __init__(
        self,
        trainer: Union[ModelWrapper, "BaalTransformersTrainer"],
        al_dataset: ActiveLearningDataset,
        eval_dataset: Dataset,
        heuristic: AbstractHeuristic,
        query_size: int = 100,
        iterations: int = 20,
        pool_size: Optional[int] = None,
        criterion: Optional[StoppingCriterion] = None,
    ):
        self.al_dataset = al_dataset
        self.eval_dataset = eval_dataset
        self.heuristic = heuristic
        self.query_size = query_size
        self.iterations = iterations
        self.criterion = criterion or LabellingBudgetStoppingCriterion(
            al_dataset, labelling_budget=al_dataset.n_unlabelled
        )
        self.pool_size = pool_size
        self.adapter = self._get_adapter(trainer)

    def start(self):
        records = []
        _start = len(self.al_dataset)
        if _start == 0:
            raise ValueError(
                "No item labelled in the training set."
                " Did you run `ActiveLearningDataset.label_randomly`?"
            )
        for _ in tqdm(
            itertools.count(start=0),  # Infinite counter to rely on Criterion
            desc="Active Experiment",
            # Upper bound estimation.
            total=np.round(self.al_dataset.n_unlabelled // self.query_size),
        ):
            self.adapter.reset_weights()
            train_metrics = self.adapter.train(self.al_dataset)
            eval_metrics = self.adapter.evaluate(
                self.eval_dataset, average_predictions=self.iterations
            )
            pool = self._get_pool()
            ranks, uncertainty = self.heuristic.get_ranks(
                self.adapter.predict(pool, iterations=self.iterations)
            )
            self.al_dataset.label(ranks[: self.query_size])
            records.append({**train_metrics, **eval_metrics})
            if self.criterion.should_stop(eval_metrics, uncertainty):
                log.info("Experiment complete", num_labelled=len(self.al_dataset) - _start)
                return records

    def _get_adapter(
        self, trainer: Union[ModelWrapper, "BaalTransformersTrainer"]
    ) -> FrameworkAdapter:
        if isinstance(trainer, ModelWrapper):
            return ModelWrapperAdapter(trainer)
        elif TRANSFORMERS_AVAILABLE and isinstance(trainer, BaalTransformersTrainer):
            return TransformersAdapter(trainer)
        raise ValueError(
            f"{type(trainer)} is not a supported trainer."
            " Baal supports ModelWrapper and BaalTransformersTrainer"
        )

    def _get_pool(self):
        if self.pool_size is None:
            return self.al_dataset.pool
        pool = self.al_dataset.pool
        if len(pool) < self.pool_size:
            return pool
        indices = np.random.choice(len(pool), min(len(pool), self.pool_size), replace=False)
        return Subset(pool, indices)
