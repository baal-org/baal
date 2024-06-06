from copy import deepcopy
from typing import Dict, cast, List, Union

from numpy._typing import NDArray

from baal.active.dataset.base import Dataset
from baal.experiments import FrameworkAdapter
from baal.transformers_trainer_wrapper import BaalTransformersTrainer


class TransformersAdapter(FrameworkAdapter):
    def __init__(self, wrapper: BaalTransformersTrainer):
        self.wrapper = wrapper
        self._init_weight = deepcopy(self.wrapper.model.state_dict())

    def reset_weights(self):
        self.wrapper.model.load_state_dict(self._init_weight)
        self.wrapper.lr_scheduler = None
        self.wrapper.optimizer = None

    def train(self, al_dataset: Dataset) -> Dict[str, float]:
        self.wrapper.train_dataset = al_dataset
        return self.wrapper.train().metrics  # type: ignore

    def predict(self, dataset: Dataset, iterations: int) -> Union[NDArray, List[NDArray]]:
        return self.wrapper.predict_on_dataset(dataset, iterations=iterations)

    def evaluate(self, dataset: Dataset, average_predictions: int) -> Dict[str, float]:
        return cast(Dict[str, float], self.wrapper.evaluate(dataset))
