# -*- coding: utf-8 -*-

from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import ast
import json




class Data:

    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str,
                 use_metadata: bool = False, 
                 metadata_id_col: List[str] = None,
                 split_ratio: float = 0.8):
        
        self.dataset = dataset

        self.user_id = user_id_col
        self.item_id = item_id_col

        self.num_items = len(self.dataset[self.item_id].unique())
        self.num_users = len(self.dataset[self.user_id].unique())

        self.dataset = self._get_negative_items(self.dataset)

        self.use_metadata = use_metadata

        self.split_ratio = split_ratio
        
        if use_metadata:
            self.metadata_id = metadata_id_col
            self.dataset = self._add_positive_metadata(self.dataset)
            self.dataset = self._add_negative_metadata(self.dataset)
            self.negative_metadata_id = self._get_negative_metadata_column_names()
            self.metadata_size = self._get_metadata_size(self.metadata_id)

    def _get_metadata_size(self, metadata_id_col):
        
        metadata_size = {}

        for col in metadata_id_col:
            metadata_size.update({col: len(self.dataset[col].unique())})

        return metadata_size
  
    def _get_negative_items(self, dataset):

        negative_items = np.random.randint(low=0, 
                                           high=self.num_items, 
                                           size=self.dataset.shape[0])

        dataset['neg_item'] = negative_items

        return dataset

    def _add_positive_metadata(self, dataset):

        dataset['pos_metadata_id'] = self.dataset[self.metadata_id].values.tolist()

        return dataset

    def _get_metadata(self, dataset):

        metadata = (dataset
                    .set_index(self.item_id)[self.metadata_id]
                    .reset_index()
                    .drop_duplicates())

        return metadata
                        
    def _add_negative_metadata(self, dataset):
        
        metadata = self._get_metadata(dataset)

        metadata_negative_names = self._get_negative_metadata_column_names()

        metadata.columns = ['neg_item'] + metadata_negative_names

        dataset = pd.merge(dataset, metadata, on='neg_item')

        dataset['neg_metadata_id'] = dataset[metadata_negative_names].values.tolist()

        return dataset

    def _get_negative_metadata_column_names(self):

        return ['neg_' + metadata_column for metadata_column in self.metadata_id]

    def _apply_negative_metadata(self):

        metadata = self._get_negative_metadata()

        dataset = pd.merge(self.dataset, metadata, on='neg_item')

        return dataset

    def map_item_metadata(self):

        grouping_col = [self.item_id] + self.metadata_id

        data = self.dataset.groupby(grouping_col, as_index=False).agg({self.user_id: 'count'}).sort_values('product_code')

        dummies = None
        for metadata in grouping_col:
            dummy = pd.get_dummies(data[metadata])
            dummies = pd.concat([dummies, dummy], axis=1) if dummies is not None else dummy

        mapping = dummies.values

        return mapping

                        
class ProcessData(Data):

    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str,
                 use_metadata: bool, 
                 metadata_id_col: List[str],
                 split_ratio: float):

        super().__init__(dataset, user_id_col, item_id_col, use_metadata, metadata_id_col, split_ratio)

        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.metadata_id_col = metadata_id_col
        self.use_metadata = use_metadata


    def _fit(self):

        cols = [self.user_id_col, self.item_id_col, 'neg_item']
        
        if self.use_metadata:
            cols += ['pos_metadata_id', 'neg_metadata_id']

        dataset = self.dataset[cols]

        new_cols = ['user_id', 'pos_item_id', 'neg_item_id']
        if self.use_metadata:
            new_cols += ['pos_metadata_id', 'neg_metadata_id']

        dataset.columns = new_cols

        if self.split_ratio < 1:

            df_train, df_test = train_test_split(dataset, test_size=1-self.split_ratio)

            return df_train, df_test

        else:
            return dataset


    def write_data(self, path: str):

        config =  {
                   'num_users': self.num_users,
                   'num_items': self.num_items,
                   'num_metadata': self.metadata_size
                  }

        with open(f'{path}/config.json', 'w') as file:
            json.dump(config, file)

        if self.split_ratio < 1:

            train, test = self._fit()

            train.to_csv(f'{path}/train.csv', index=False)
            test.to_csv(f'{path}/test.csv', index=False)

        else:
            train = self._fit()    
            train.to_csv(f'{path}/train.csv', index=False)             


class FastDataLoader:
    
    def __init__(self, 
                 path, 
                 use_metadata, 
                 batch_size=32):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        if use_metadata:
            self.tensors = pd.read_csv(path,
                                       converters={'pos_metadata_id': eval,
                                                   'neg_metadata_id': eval}, 
                                       chunksize=batch_size)

        else:
            self.tensors = pd.read_csv(path, chunksize=batch_size)
                
        self.dataset_len = batch_size
        self.batch_size = batch_size

        self.use_metadata = use_metadata
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        
        if self.i >= self.dataset_len:
            raise StopIteration
        
        batch = self.tensors.get_chunk().reset_index()
        
        self.i += self.batch_size
        self.dataset_len += batch.shape[0]
        
        if self.use_metadata:

            return  {
                    'user_id': torch.from_numpy(batch['user_id'].values),
                    'pos_item_id': torch.from_numpy(batch['pos_item_id'].values),
                    'neg_item_id': torch.from_numpy(batch['neg_item_id'].values),
                    'pos_metadata_id': torch.Tensor(batch['pos_metadata_id']).long(),
                    'neg_metadata_id': torch.Tensor(batch['neg_metadata_id']).long()
                    }

        else:
                
            return  {
                    'user_id': torch.from_numpy(batch['user_id'].values),
                    'pos_item_id': torch.from_numpy(batch['pos_item_id'].values),
                    'neg_item_id': torch.from_numpy(batch['neg_item'].values)
                    }