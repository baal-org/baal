from copy import deepcopy
from typing import Dict, Union, List

from numpy._typing import NDArray

from baal import ModelWrapper
from baal.active.dataset.base import Dataset
from baal.experiments import FrameworkAdapter


class ModelWrapperAdapter(FrameworkAdapter):
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper
        self._init_weight = deepcopy(self.wrapper.state_dict())

    def reset_weights(self):
        self.wrapper.load_state_dict(self._init_weight)

    def train(self, al_dataset: Dataset) -> Dict[str, float]:
        self.wrapper.train_on_dataset(al_dataset)
        return self.wrapper.get_metrics("train")

    def predict(self, dataset: Dataset, iterations: int) -> Union[NDArray, List[NDArray]]:
        return self.wrapper.predict_on_dataset(dataset, iterations=iterations)

    def evaluate(self, dataset: Dataset, average_predictions: int) -> Dict[str, float]:
        self.wrapper.test_on_dataset(dataset, average_predictions=average_predictions)
        return self.wrapper.get_metrics("test")
