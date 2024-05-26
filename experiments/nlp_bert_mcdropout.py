import argparse
import random
from copy import deepcopy
from pprint import pprint

import torch
import torch.backends
from tqdm import tqdm

# These packages are optional and not needed for BaaL main package.
# You can have access to `datasets` and `transformers` if you install
# BaaL with --dev setup.
from datasets import load_dataset
from transformers import BertTokenizer, TrainingArguments
from transformers import BertForSequenceClassification

from baal.active import get_heuristic
from baal.active.active_loop import ActiveLearningLoop
from baal.active.dataset.nlp_datasets import active_huggingface_dataset, HuggingFaceDatasets
from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module
from baal.experiments.base import ActiveLearningExperiment
from baal.transformers_trainer_wrapper import BaalTransformersTrainer

"""
Minimal example to use BaaL for NLP Classification.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--model", default="bert-base-uncased", type=str)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--learning_epoch", default=20, type=int)
    return parser.parse_args()


def get_datasets(initial_pool, tokenizer):

    # To be able to support most cases, we have provided support for
    # HuggingFace datasets. You can always create a custom wrapper for
    # custom datasets.
    datasets = load_dataset("glue", "sst2")
    raw_train_set = datasets["train"]
    raw_valid_set = datasets["validation"]

    active_set = active_huggingface_dataset(raw_train_set, tokenizer)
    active_set.can_label = False
    valid_set = HuggingFaceDatasets(raw_valid_set, tokenizer)

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, valid_set


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)

    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model"]
    )
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=hyperparams["model"])

    # In this example we use tokenizer once only in the beginning since it would
    # make the whole process faster. However, it is also possible to input tokenizer
    # in trainer.
    active_set, test_set = get_datasets(hyperparams["initial_pool"], tokenizer)

    # change dropout layer to MCDropout
    model = patch_module(model)

    if use_cuda:
        model.cuda()

    training_args = TrainingArguments(
        output_dir="/app/baal/results",  # output directory
        num_train_epochs=hyperparams["learning_epoch"],  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        weight_decay=0.01,  # strength of weight decay
        logging_dir="/app/baal/logs",  # directory for storing logs
    )

    # We wrap the huggingface Trainer to create an Active Learning Trainer
    model = BaalTransformersTrainer(
        model=model,
        args=training_args,
        train_dataset=active_set,
        eval_dataset=test_set,
        tokenizer=None,
    )
    experiment = ActiveLearningExperiment(
        trainer=model,
        al_dataset=active_set,
        eval_dataset=test_set,
        heuristic=BALD(),
        query_size=hyperparams["query_size"],
        iterations=20,
        criterion=None,
    )
    pprint(experiment.start())


if __name__ == "__main__":
    main()
