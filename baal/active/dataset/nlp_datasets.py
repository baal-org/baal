from typing import List

import numpy as np
import torch
from baal.active.dataset.base import Dataset
from datasets import Dataset as HFDataset

from baal.active import ActiveLearningDataset


class HuggingFaceDatasets(Dataset):
    """
    Support for `huggingface.datasets`: (https://github.com/huggingface/datasets).
    The purpose of this wrapper is to separate the labels from the rest of the sample information
    and make the dataset ready to be used by `baal.active.ActiveLearningDataset`.

    Args:
        dataset (Dataset): a dataset provided by huggingface.
        tokenizer (transformers.PreTrainedTokenizer): a tokenizer provided by huggingface.
        target_key (str): target key used in the dataset's dictionary.
        input_key (str): input key used in the dataset's dictionary.
        max_seq_len (int): max length of a sequence to be used for padding the shorter
            sequences.
    """

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer=None,
        target_key: str = "label",
        input_key: str = "sentence",
        max_seq_len: int = 128,
    ):
        self.dataset = dataset
        self.targets, self.texts = self.dataset[target_key], self.dataset[input_key]
        self.targets_list: List = np.unique(self.targets).tolist()
        self.input_ids, self.attention_masks = (
            self._tokenize(tokenizer, max_seq_len) if tokenizer else ([], [])
        )

    @property
    def num_classes(self):
        return len(self.targets_list)

    def _tokenize(self, tokenizer, max_seq_len):
        # For speed purposes, we should use fast tokenizers here, but that is up to the caller
        tokenized = tokenizer(
            self.texts,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def label(self, idx: int, value: int):
        """Label the item.

        Args:
            idx: index to label
            value: Value to label the index.
        """
        self.targets[idx] = value

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        target = self.targets_list.index(self.targets[idx])

        return {
            "input_ids": self.input_ids[idx].flatten() if len(self.input_ids) > 0 else None,
            "inputs": self.texts[idx],
            "attention_mask": self.attention_masks[idx].flatten()
            if len(self.attention_masks) > 0
            else None,
            "label": torch.tensor(target, dtype=torch.long),
        }


def active_huggingface_dataset(
    dataset,
    tokenizer=None,
    target_key: str = "label",
    input_key: str = "sentence",
    max_seq_len: int = 128,
    **kwargs
):
    """
    Wrapping huggingface.datasets with baal.active.ActiveLearningDataset.

    Args:
        dataset (torch.utils.data.Dataset): a dataset provided by huggingface.
        tokenizer (transformers.PreTrainedTokenizer): a tokenizer provided by huggingface.
        target_key (str): target key used in the dataset's dictionary.
        input_key (str): input key used in the dataset's dictionary.
        max_seq_len (int): max length of a sequence to be used for padding the shorter sequences.
        kwargs (Dict): Parameters forwarded to 'ActiveLearningDataset'.

    Returns:
        an baal.active.ActiveLearningDataset object.
    """

    return ActiveLearningDataset(
        HuggingFaceDatasets(dataset, tokenizer, target_key, input_key, max_seq_len), **kwargs
    )
