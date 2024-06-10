import abc
from typing import Dict, List, Union

from numpy._typing import NDArray

from baal.active.dataset.base import Dataset


class FrameworkAdapter(abc.ABC):
    def reset_weights(self):
        raise NotImplementedError

    def train(self, al_dataset: Dataset) -> Dict[str, float]:
        raise NotImplementedError

    def predict(self, dataset: Dataset, iterations: int) -> Union[NDArray, List[NDArray]]:
        raise NotImplementedError

    def evaluate(self, dataset: Dataset, average_predictions: int) -> Dict[str, float]:
        raise NotImplementedError
