import argparse
import random
from copy import deepcopy

import torch
import torch.backends
from tqdm import tqdm

# These packages are optional and not needed for BaaL main package.
# You can have access to `dataset` and `transformers` if you install
# BaaL with --dev setup.
from datasets import load_dataset
from transformers import BertTokenizer, TrainingArguments
from transformers import BertForSequenceClassification

from baal import get_heuristic
from baal import active_huggingface_dataset, HuggingFaceDatasets
from baal import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.transformers_trainer_wrapper import BaalTransformersTrainer

"""
Minimal example to use BaaL for NLP Classification.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--model", default="bert-base-uncased", type=str)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=20, type=int)
    return parser.parse_args()


def get_datasets(initial_pool, tokenizer):

    # To be able to support most cases, we have provided support for
    # HuggingFace dataset. You can always create a custom wrapper for
    # custom dataset.
    datasets = load_dataset("glue", "sst2")
    raw_train_set = datasets["train"]
    raw_valid_set = datasets["validation"]

    active_set = active_huggingface_dataset(raw_train_set, tokenizer)
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

    heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])

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

    init_weights = deepcopy(model.state_dict())

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

    logs = {}
    logs["epoch"] = 0

    # In this case, nlp data is fast to process and we do NoT need to use a smaller batch_size
    active_loop = ActiveLearningLoop(
        active_set,
        model.predict_on_dataset,
        heuristic,
        hyperparams.get("query_size", 1),
        iterations=hyperparams["iterations"],
    )

    for epoch in tqdm(range(args.epoch)):
        # we use the default setup of HuggingFace for training (ex: epoch=1).
        # The setup is adjustable when BaalHuggingFaceTrainer is defined.
        model.train()

        # Validation!
        eval_metrics = model.evaluate()

        # We reorder the unlabelled pool at the frequency of learning_epoch
        # This helps with speed while not changing the quality of uncertainty estimation.
        should_continue = active_loop.step()

        # We reset the model weights to relearn from the new trainset.
        model.load_state_dict(init_weights)
        model.lr_scheduler = None
        if not should_continue:
            break
        active_logs = {
            "epoch": epoch,
            "labeled_data": active_set._labelled,
            "Next Training set size": len(active_set),
        }

        logs = {**eval_metrics, **active_logs}
        print(logs)


if __name__ == "__main__":
    main()
