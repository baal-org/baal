import unittest
import pytest
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from baal.active.dataset import ActiveLearningDataset
from baal.active.dataset.nlp_datasets import HuggingFaceDatasets
import datasets


class MyDataset(Dataset):
    def __init__(self):
        self.dataset = datasets.Dataset.from_dict({'sentence': [f'this is test number {i}' for i in range(10)],
                                                   'label': ['POS' if (i % 2) == 0 else 'NEG' for i in range(10)]},
                                                  features=datasets.Features({'sentence': datasets.Value('string'),
                                                                              'label': datasets.ClassLabel(2,
                                                                                                           names=['NEG',
                                                                                                                  'POS'])}))

    def __len__(self):
        return 10

    def __getitem__(self, item):

        if isinstance(item, int):
            return {'sentence': self.dataset['sentence'][item],
                    'label': self.dataset['label'][item]}
        elif isinstance(item, str):
            return self.dataset[item]


class ActiveArrowDatasetTest(unittest.TestCase):
    def setUp(self):
        dataset = datasets.Dataset.from_dict({'sentence': [f'this is test number {i}' for i in range(10)],
                                              'label': ['POS' if (i % 2) == 0 else 'NEG' for i in range(10)]},
                                             features=datasets.Features({'sentence': datasets.Value('string'),
                                                                         'label': datasets.ClassLabel(2, names=['NEG',
                                                                                                                'POS'])}))
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        def preprocess(example):
            results = tokenizer(example['sentence'], max_length=50,
                                truncation=True, padding='max_length')
            return results

        tokenized_dataset = dataset.map(preprocess, batched=True)
        self.active_dataset = ActiveLearningDataset(tokenized_dataset)

    def test_dataset(self):
        assert len(self.active_dataset) == 0
        self.active_dataset.label_randomly(2)
        assert len(self.active_dataset) == 2


class HuggingFaceDatasetsTest(unittest.TestCase):
    def setUp(self):
        dataset = MyDataset()
        self.dataset = HuggingFaceDatasets(dataset)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dataset_with_tokenizer = HuggingFaceDatasets(dataset, tokenizer=tokenizer)

    def test_dataset(self):
        assert len(self.dataset) == len(self.dataset_with_tokenizer) == 10
        assert [key in ['inputs', 'input_ids', 'attention_mask', 'label'] for key, value
                in self.dataset[0].items()]
        assert self.dataset[0]['inputs'] == 'this is test number 0'
        assert self.dataset[0]['label'] == 1
        assert self.dataset[0]['input_ids'] is None
        assert self.dataset[0]['attention_mask'] is None
        assert self.dataset.num_classes == 2

    def test_tokenizer(self):
        assert [key in ['inputs', 'input_ids', 'attention_mask', 'label'] for key, value
                in self.dataset_with_tokenizer[0].items()]
        assert self.dataset_with_tokenizer[0]['inputs'] == 'this is test number 0'
        assert self.dataset_with_tokenizer[0]['label'] == 1
        assert self.dataset_with_tokenizer.input_ids.shape[1] <= 128
        assert len(self.dataset_with_tokenizer[0]['attention_mask']) == \
               len(self.dataset_with_tokenizer[0]['input_ids'])

    def test_label(self):
        prev_label = self.dataset[2]['label'].item()
        self.dataset.label(2, (prev_label + 1) % 2)
        assert self.dataset[2]['label'].item() == ((prev_label - 1) % 2)


if __name__ == '__main__':
    pytest.main()
