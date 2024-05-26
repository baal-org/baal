import itertools
from typing import Union, Optional, TYPE_CHECKING

import pandas as pd
import structlog
from tqdm import tqdm

from baal import ModelWrapper, ActiveLearningDataset
from baal.active.dataset.base import Dataset
from baal.active.heuristics import AbstractHeuristic
from baal.active.stopping_criteria import StoppingCriterion, LabellingBudgetStoppingCriterion
from baal.experiments import FrameworkAdapter
from baal.experiments.modelwrapper import ModelWrapperAdapter

try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    from baal.transformers_trainer_wrapper import BaalTransformersTrainer
    from baal.experiments.transformers import TransformersAdapter

    TRANSFORMERS_AVAILABLE = False

log = structlog.get_logger(__name__)


class ActiveLearningExperiment:
    def __init__(
        self,
        trainer: Union[ModelWrapper, "BaalTransformersTrainer"],
        al_dataset: ActiveLearningDataset,
        eval_dataset: Dataset,
        heuristic: AbstractHeuristic,
        query_size: int = 100,
        iterations: int = 20,
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
        self.adapter = self._get_adapter(trainer)

    def start(self):
        records = []
        _start = len(self.al_dataset)
        for _ in tqdm(itertools.count(start=0)):
            self.adapter.reset_weights()
            train_metrics = self.adapter.train(self.al_dataset)
            eval_metrics = self.adapter.evaluate(
                self.eval_dataset, average_predictions=self.iterations
            )
            ranks, uncertainty = self.heuristic.get_ranks(
                self.adapter.predict(self.al_dataset.pool, iterations=self.iterations)
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
