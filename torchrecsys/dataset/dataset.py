# -*- coding: utf-8 -*-

from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class DataFarm():

    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str,
                 use_metadata: bool, 
                 metadata_id_col: List[str]):
        
        self.dataset = dataset

        self.user_id = user_id_col
        self.item_id = item_id_col
        self.metadata_id = metadata_id_col

        self.num_items = len(self.dataset[self.item_id].unique())
        self.num_users = len(self.dataset[self.user_id].unique())

        self.dataset = self._get_negative_items(self.dataset)

        self.use_metadata = use_metadata
        
        if use_metadata:
            self.dataset = self._get_negative_metadata(self.dataset)
            self.negative_metadata_id = self._get_negative_metadata_column_names()
  
    def _get_negative_items(self, dataset):

        negative_items = np.random.randint(low=0, 
                                           high=self.num_items, 
                                           size=self.dataset.shape[0])

        dataset['neg_item'] = negative_items

        return dataset

    def _get_metadata(self):

        metadata = (self.dataset
                    .set_index(self.item_id)[self.metadata_id]
                    .reset_index()
                    .drop_duplicates())

        return metadata
                        
    def _get_negative_metadata(self, dataset):
        
        metadata = self._get_metadata()

        metadata_negative_names = self._get_negative_metadata_column_names()

        metadata.columns = ['neg_item'] + metadata_negative_names

        dataset = pd.merge(dataset, metadata, on='neg_item')

        return dataset

    def _get_negative_metadata_column_names(self):

        return ['neg_' + metadata_column for metadata_column in self.metadata_id]

    def _apply_negative_metadata(self):

        metadata = self._get_negative_metadata()

        dataset = pd.merge(self.dataset, metadata, on='neg_item')

        return dataset


class CustomDataset(Dataset, DataFarm):

    """
    Class for loading data from a pandas DataFrame, containing user-product interactions/ratings/purchases
    plus item side information like product name, product category, etc.

    It returns a PyTorch Dataset object, which can be used to load data in a PyTorch model.

    In the __getitem__ method, the dataset is loaded in batches, and the data is returned as a dictionary.

    Also negative items are sampled from the same dataset, and returned as a dictionary along with their metadata.

    :param dataset: pd.DataFrame
    :param user_id_col: str
    :param item_id_col: str
    :param metadata_id_col: list of str
    """

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):

        return {'user_id': self.dataset.iloc[idx][self.user_id],
                'pos_item_id': self.dataset.iloc[idx][self.item_id],
                'pos_metadata_id': [metadata for metadata in self.dataset.iloc[idx][self.metadata_id]],
                'neg_item_id': self.dataset.iloc[idx]['neg_item'],
                'neg_metadata_id': [metadata for metadata in self.dataset.iloc[idx][self.negative_metadata_id]]
                }
              

class CustomDataLoader(DataFarm):

    def fit(self):

        if self.use_metadata:

            return {'user_id': torch.from_numpy(self.dataset[self.user_id].values),
                    'pos_item_id': torch.from_numpy(self.dataset[self.item_id].values),
                    'pos_metadata_id': [torch.from_numpy(self.dataset[self.metadata_id].values)],
                    'neg_item_id': torch.from_numpy(self.dataset['neg_item'].values),
                    'neg_metadata_id': [torch.from_numpy(self.dataset[self.negative_metadata_id].values)]
                    }

        else: 

            return {'user_id': torch.from_numpy(self.dataset[self.user_id].values),
                    'pos_item_id': torch.from_numpy(self.dataset[self.item_id].values),
                    'neg_item_id': torch.from_numpy(self.dataset['neg_item'].values),
                    }

