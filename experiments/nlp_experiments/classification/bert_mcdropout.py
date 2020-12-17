import sys
sys.path.append("/app/baal")
import argparse
import random

import torch
import torch.backends
from torch import optim
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from tqdm import tqdm

from baal.utils.cuda_utils import to_cuda
from baal.utils.iterutils import map_on_tensor
from baal.utils.array_utils import stack_in_memory
from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import ModelWrapper
from experiments.nlp_experiments.classification import CSVClassificationDataset

"""
Minimal example to use BaaL for NLP Classification.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--n_data_to_label", default=100, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument('--learning_epoch', default=20, type=int)
    return parser.parse_args()

def get_datasets(initial_pool):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
    train_ds = CSVClassificationDataset(
        folder='/app/baal/experiments/nlp_experiments/classification/data',
        input_key='text', target_key='label', tokenizer=tokenizer, split='train')
    test_set = CSVClassificationDataset(
        folder='/app/baal/experiments/nlp_experiments/classification/data',
        input_key='text', target_key='label', tokenizer=tokenizer, split='test')

    active_set = ActiveLearningDataset(train_ds)

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set


class BertWrapper(ModelWrapper):
    def __init__(self, model):
        super(BertWrapper, self).__init__(model=model, criterion=None, replicate_in_memory=True)
        self.model = patch_module(model)

    def train_on_batch(self, data, target, optimizer, cuda=False, regularizer=None):

        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        if cuda:
            input_ids, attention_mask, target = to_cuda(input_ids), to_cuda(attention_mask),\
                                                to_cuda(target)
        optimizer.zero_grad()
        loss, text_fea = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=target)[:2]

        loss.backward()
        optimizer.step()
        self._update_metrics(text_fea, target, loss, filter='train')
        return loss

    def test_on_batch(
            self,
            data: dict(),
            target: torch.Tensor,
            cuda: bool = False,
            average_predictions: int = 1
    ):
        with torch.no_grad():
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            if cuda:
                input_ids, attention_mask, target = to_cuda(input_ids), to_cuda(attention_mask),\
                                                    to_cuda(target)
            loss, text_fea = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        labels=target)[:2]
            self._update_metrics(text_fea, target, loss, 'test')
            return loss

    def predict_on_batch(self, data, iterations=1, cuda=False):

        # TODO: do we need replicate in memory = False for NLP?
        with torch.no_grad():
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            if cuda:
                input_ids, attention_mask = to_cuda(input_ids), to_cuda(attention_mask)
            if self.replicate_in_memory:
                input_ids = map_on_tensor(lambda d: stack_in_memory(d, iterations), input_ids)
                attention_mask = map_on_tensor(lambda d: stack_in_memory(d, iterations),
                                               attention_mask)
                try:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
                except RuntimeError as e:
                    raise RuntimeError(
                        '''CUDA ran out of memory while BaaL tried to replicate data.
                        See the exception above. Use `replicate_in_memory=False` in order
                        to reduce the memory requirements.
                        Note that there will be some speed trade-offs''') from e
                out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
                out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
            return out


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)

    active_set, test_set = get_datasets(hyperparams['initial_pool'])

    heuristic = get_heuristic(hyperparams['heuristic'],
                              hyperparams['shuffle_prop'])

    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased')

    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Wraps the model into a usable API.
    model = BertWrapper(model)

    logs = {}
    logs['epoch'] = 0

    # for prediction we use a smaller batchsize
    # since it is slower
    active_loop = ActiveLearningLoop(active_set,
                                     model.predict_on_dataset,
                                     heuristic,
                                     hyperparams.get('n_data_to_label', 1),
                                     batch_size=10,
                                     iterations=hyperparams['iterations'],
                                     use_cuda=use_cuda)

    for epoch in tqdm(range(args.epoch)):
        model.train_on_dataset(active_set, optimizer, hyperparams["batch_size"],
                               epoch=1, use_cuda=use_cuda)

        # Validation!
        model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda=use_cuda)
        metrics = model.metrics

        if epoch % hyperparams['learning_epoch'] == 0:
            should_continue = active_loop.step()
            model.reset_fcs()
            if not should_continue:
                break
        val_loss = metrics['test_loss'].value
        logs = {
            "val": val_loss,
            "epoch": epoch,
            "train": metrics['train_loss'].value,
            "labeled_data": active_set._labelled,
            "Next Training set size": len(active_set)
        }
        print(logs)


if __name__ == "__main__":
    main()
