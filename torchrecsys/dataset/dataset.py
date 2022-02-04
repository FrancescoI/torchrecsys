# -*- coding: utf-8 -*-

from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
from torchrecsys.helper.cuda import gpu, cpu

class CustomDataset(Dataset):

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
    
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str, 
                 metadata_id_col: List[str]):
        
        self.dataset = dataset

        self.user_id = user_id_col
        self.item_id = item_id_col
        self.metadata_id = metadata_id_col

        self.num_items = len(self.dataset[self.item_id].unique())
        self.num_users = len(self.dataset[self.user_id].unique())

        self.dataset['negative_item'] = self._get_negative_items()
        self.dataset = self._apply_negative_metadata()

        self.negative_metadata_id = self._get_negative_metadata_column_names()

    def __len__(self):
        return self.dataset.shape[0]
    
    def _get_negative_items(self):

        negative_items = np.random.randint(low=0, high=self.num_items, size=self.dataset.shape[0]) 

        return negative_items

    def _get_metadata(self):

        metadata = (self.dataset
                    .set_index(self.item_id)[self.metadata_id]
                    .reset_index()
                    .drop_duplicates())

        return metadata
                        
    def _get_negative_metadata(self):
        
        metadata = self._get_metadata()

        metadata_negative_names = self._get_negative_metadata_column_names()

        metadata.columns = ['negative_item'] + metadata_negative_names

        return metadata

    def _get_negative_metadata_column_names(self):

        return ['neg_' + metadata_column for metadata_column in self.metadata_id]

    def _apply_negative_metadata(self):

        metadata = self._get_negative_metadata()

        dataset = pd.merge(self.dataset, metadata, on='negative_item')

        return dataset

    def __getitem__(self, idx):
        
        return {'user_id': self.dataset.iloc[idx][self.user_id],
                'positive_item_id': {
                    'item_id': self.dataset.iloc[idx][self.item_id],
                    'metadata_id': {key: value for 
                                    key, value in zip(self.metadata_id, 
                                                      self.dataset.iloc[idx][self.metadata_id])}
                            },
                'negative_item_id': {
                    'item_id': self.dataset.iloc[idx]['negative_item'],
                    'metadata_id': {key: value for 
                                    key, value in zip(self.metadata_id, 
                                                      self.dataset.iloc[idx][self.negative_metadata_id])}
                    }
                }
              
class CustomDataLoader:

    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str, 
                 metadata_id_col: List[str],
                 use_cuda: bool = False):
        
        self.dataset = dataset

        self.user_id = user_id_col
        self.item_id = item_id_col
        self.metadata_id = metadata_id_col

        self.use_cuda = use_cuda

        self.num_items = len(self.dataset[self.item_id].unique())
        self.num_users = len(self.dataset[self.user_id].unique())

        self.negative_metadata_id = self._get_negative_metadata_column_names()
    
    def _get_negative_items(self):

        negative_items = np.random.randint(low=0, high=self.num_items, size=self.dataset.shape[0]) 

        return negative_items

    def _get_metadata(self):

        metadata = (self.dataset
                    .set_index(self.item_id)[self.metadata_id]
                    .reset_index()
                    .drop_duplicates())

        return metadata
                        
    def _get_negative_metadata(self):
        
        metadata = self._get_metadata()

        metadata_negative_names = self._get_negative_metadata_column_names()

        metadata.columns = ['negative_item'] + metadata_negative_names

        return metadata

    def _get_negative_metadata_column_names(self):

        return ['neg_' + metadata_column for metadata_column in self.metadata_id]

    def _apply_negative_metadata(self):

        metadata = self._get_negative_metadata()

        dataset = pd.merge(self.dataset, metadata, on='negative_item')

        return dataset

    def fit(self):
        
        self.dataset['negative_item'] = self._get_negative_items()
        self.dataset = self._apply_negative_metadata()

        return {'user_id': gpu(torch.from_numpy(self.dataset[self.user_id].values), self.use_cuda),
                'positive_item_id': {
                    'item_id': gpu(torch.from_numpy(self.dataset[self.item_id].values), self.use_cuda),
                    'metadata_id': {key: gpu(torch.from_numpy(value), self.use_cuda) for 
                                    key, value in zip(self.metadata_id, 
                                                      self.dataset[self.metadata_id].values)}
                            },
                'negative_item_id': {
                    'item_id': gpu(torch.from_numpy(self.dataset['negative_item'].values), self.use_cuda),
                    'metadata_id': {key: gpu(torch.from_numpy(value), self.use_cuda) for 
                                    key, value in zip(self.metadata_id, 
                                                      self.dataset[self.negative_metadata_id].values)}
                    }
                }