import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

pjoin = os.path.join


class CSVClassificationDataset(Dataset):
    def __init__(self, folder, input_key, target_key, tokenizer, split, max_seq_len=512, seed=1337):

        df = pd.concat([pd.read_csv(p) for p in glob(pjoin(folder, '*.csv'))], ignore_index=True)
        self.classes = df[target_key].unique().tolist()
        df = self._split_df(df, split, seed)
        self.targets = df[target_key].apply(lambda r: self.classes.index(r)).to_numpy()
        self.inputs = df[input_key].to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_seq_len

    def __repr__(self):
        labels = list(zip(*np.unique(self.targets, return_counts=True)))
        return f"CSVClassificationDataset: {labels}"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        review = str(self.inputs[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {'input_ids': encoding['input_ids'].flatten(),
                'inputs': review,
                'attention_mask': encoding['attention_mask'].flatten()}, \
               torch.tensor(target, dtype=torch.long)

    def _split_df(self, df, split, seed):
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=seed)
        if split == 'train':
            return df_train
        elif split == 'val':
            return df_val
        elif split == 'test':
            return df_test
        else:
            raise ValueError(f'{split} is not a valid split')