import unittest
import pytest
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from baal.active.nlp_datasets import HuggingFaceDatasets


class MyDataset(Dataset):
    def __init__(self):
        self.dataset = {'sentence': [],
                        'label': []}
        for i in range(10):
            self.dataset['sentence'].append(f'this is test number {i}')
            self.dataset['label'].append(0 if (i // 2) == 0 else 1)

    def __len__(self):
        return 10

    def __getitem__(self, item):

        if isinstance(item, int):
            return {'sentence': self.dataset['sentence'][item],
                    'label': self.dataset['label'][item] }
        elif isinstance(item, str):
            return self.dataset[item]


class HuggingFaceDatasetsTest(unittest.TestCase):
    def setUp(self):
        dataset = MyDataset()
        self.dataset = HuggingFaceDatasets(dataset)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dataset_with_tokenizer = HuggingFaceDatasets(dataset, tokenizer=tokenizer)

    def test_dataset(self):
        assert len(self.dataset) == len(self.dataset_with_tokenizer) == 10
        print(self.dataset[0])
        assert [key in ['inputs', 'input_ids', 'attention_mask', 'label'] for key, value
                in self.dataset[0].items()]
        assert self.dataset[0]['inputs'] == 'this is test number 0'
        assert self.dataset[0]['label'] == 0
        assert self.dataset[0]['input_ids'] is None
        assert self.dataset[0]['attention_mask'] is None

    def test_tokenizer(self):
        assert [key in ['inputs', 'input_ids', 'attention_mask', 'label'] for key, value
                in self.dataset_with_tokenizer[0].items()]
        assert self.dataset_with_tokenizer[0]['inputs'] == 'this is test number 0'
        assert self.dataset_with_tokenizer[0]['label'] == 0
        assert self.dataset_with_tokenizer.input_ids.shape[1] <= 128
        assert len(self.dataset_with_tokenizer[0]['attention_mask']) ==\
               len(self.dataset_with_tokenizer[0]['input_ids'])


if __name__ == '__main__':
    pytest.main()
