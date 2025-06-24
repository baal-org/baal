import unittest
from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import TrainingArguments

from baal.transformers_trainer_wrapper import BaalTransformersTrainer


# the DummyDataset is taken from
# (https://github.com/huggingface/transformers/blob/master/tests/test_trainer.py#L81)
class DummyDataset:
    def __init__(self, x):
        self.x = x
        self.length = 5

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}


class BaalTransformerTrainer(unittest.TestCase):
    def setUp(self):
        x = torch.randint(0, 5, (10,))
        self.dataset = DummyDataset(x)
        self.args = TrainingArguments(".")
        config = GPT2Config(vocab_size=100, n_positions=128, n_ctx=128, n_embd=32, n_layer=3,
                            n_head=4)
        self.model = GPT2LMHeadModel(config)

    def test_predict_on_dataset_generator(self):
        for replicate in [True, False]:
            wrapper = BaalTransformersTrainer(model=self.model,
                                              args=self.args,
                                              replicate_in_memory=replicate,
                                              train_dataset=self.dataset,
                                              eval_dataset=self.dataset,
                                              tokenizer=None)
            # iteration == 1
            pred = wrapper.predict_on_dataset_generator(self.dataset, 1, False)
            assert next(pred).shape == (5, 100, 10, 1)

            # iterations > 1
            pred = wrapper.predict_on_dataset_generator(self.dataset, 10, False)
            assert next(pred).shape == (5, 100, 10, 10)

            # Test generators
            l_gen = wrapper.predict_on_dataset_generator(self.dataset, 10, False)
            l = wrapper.predict_on_dataset(self.dataset, 10, False)
            assert np.allclose(next(l_gen)[0], l[0])

            # Test Half
            l_gen = wrapper.predict_on_dataset_generator(self.dataset, 10, half=True)
            l = wrapper.predict_on_dataset(self.dataset, 10, half=True)
            assert next(l_gen).dtype == np.float16
            assert l.dtype == np.float16

    def test_load_state_dict(self):
        wrapper = BaalTransformersTrainer(model=self.model,
                                          args=self.args,
                                          train_dataset=self.dataset,
                                          eval_dataset=self.dataset,
                                          tokenizer=None)
        model_weights = deepcopy(list(wrapper.model.parameters())[0])
        initial_state_dict = deepcopy(wrapper.model.state_dict())
        wrapper.train()
        weights_after_training = deepcopy(list(wrapper.model.parameters())[0])
        assert not torch.equal(model_weights.data, weights_after_training.data)

        wrapper.load_state_dict(initial_state_dict)

        reloaded_weights = deepcopy(list(wrapper.model.parameters())[0])
        assert torch.equal(model_weights.data, reloaded_weights.data)


if __name__ == '__main__':
    pytest.main()
