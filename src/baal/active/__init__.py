from typing import Union, Callable, Dict

from . import heuristics
from .active_loop import ActiveLearningLoop
from .dataset import ActiveLearningDataset
from .file_dataset import FileDataset
from .nlp_datasets import HuggingFaceDatasets


def get_heuristic(name: str, shuffle_prop: float = 0.0,
                  reduction: Union[str, Callable] = 'none',
                  **kwargs) -> heuristics.AbstractHeuristic:
    """
    Create an heuristic object from the name.

    Args:
        name (str): Name of the heuristic.
        shuffle_prop (float): Shuffling proportion when getting ranks.
        reduction (Union[str, Callable]): Reduction used after computing the score.
        kwargs (dict): Complementary arguments.

    Returns:
        AbstractHeuristic object.
    """
    return {
        'random': heuristics.Random,
        'certainty': heuristics.Certainty,
        'entropy': heuristics.Entropy,
        'margin': heuristics.Margin,
        'bald': heuristics.BALD,
        'variance': heuristics.Variance,
        'precomputed': heuristics.Precomputed,
        'batch_bald': heuristics.BatchBALD
    }[name](shuffle_prop=shuffle_prop, reduction=reduction, **kwargs)


def active_huggingface_dataset(dataset,
                               tokenizer=None,
                               target_key: str = "label",
                               input_key: str = "sentence",
                               max_seq_len: int = 128,
                               **kwargs):
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

    return ActiveLearningDataset(HuggingFaceDatasets(dataset,
                                                     tokenizer,
                                                     target_key,
                                                     input_key,
                                                     max_seq_len), **kwargs)
