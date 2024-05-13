from typing import Dict

from numpy._typing import NDArray

from baal import ModelWrapper
from baal.active.dataset.base import Dataset
from baal.experiments import FrameworkAdapter


class ModelWrapperAdapter(FrameworkAdapter):
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper
        self._init_weight = self.wrapper.state_dict()

    def reset_weights(self):
        self.wrapper.load_state_dict(self._init_weight)

    def train(self, al_dataset: Dataset) -> Dict[str, float]:
        # TODO figure out args. Probably add a new "TrainingArgs" similar to other framworks.
        return self.wrapper.train_on_dataset(al_dataset,
                                             optimizer=...,
                                             batch_size=...,
                                             epoch=...,
                                             use_cuda=...,
                                             workers=...,
                                             collate_fn=...,
                                             regularizer=...)

    def predict(self, dataset: Dataset, iterations: int) -> NDArray:
        return self.wrapper.predict_on_dataset(dataset,
                                               iterations=iterations,
                                               batch_size=...,
                                               use_cuda=...,
                                               workers=...,
                                               collate_fn=..., )

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        return self.wrapper.test_on_dataset(dataset,
                                            batch_size=...,
                                            use_cuda=...,
                                            workers=...,
                                            collate_fn=...)
