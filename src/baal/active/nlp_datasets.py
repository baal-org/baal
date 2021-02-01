from typing import Optional, Callable, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from baal.active import ActiveLearningDataset

def _identity(x):
    return x

def active_dataset(dataset,
                   tokenizer,
                   target_key: str = "label",
                   input_key: str = "sentence",
                   max_seq_len: int = 128,
                   eval_transform: Optional[Callable] = None,
                   labelled: Union[np.ndarray, torch.Tensor] = None,
                   make_unlabelled: Callable = _identity,
                   random_state=None,
                   pool_specifics: Optional[dict] = None):

    return ActiveLearningDataset(HuggingFaceDatasets(dataset,
                                                     tokenizer,
                                                     target_key,
                                                     input_key,
                                                     max_seq_len),
                                 eval_transform,
                                 labelled,
                                 make_unlabelled,
                                 random_state,
                                 pool_specifics)


class HuggingFaceDatasets(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 target_key: str = "label",
                 input_key: str = "sentence",
                 max_seq_len: int = 128,
                 ):
        """
        Support for HuggingFace Datasets
        Args:
            dataset:
            tokenizer:
            max_seq_len:
        """
        self.dataset = dataset

        # self.classes = [0, 1]
        self.targets, self.texts = self.dataset[target_key], self.dataset[input_key]
        self.input_ids, self.attention_masks = self._tokenize(tokenizer, max_seq_len)

    def _tokenize(self, tokenizer, max_seq_len):
        # For speed purposes, we should use fast tokenizers here, but that is up to the caller
        tokenized = tokenizer(
            self.texts,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return tokenized["input_ids"], tokenized["attention_mask"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        target = self.targets[idx]

        return (
            {
                'input_ids': self.input_ids[idx].flatten(),
                'inputs': self.texts[idx],
                'attention_mask': self.attention_masks[idx].flatten(),
            },
            torch.tensor(target, dtype=torch.long),
        )


