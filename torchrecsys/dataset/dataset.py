# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset

class XXX():
    
    def __init__(self, brand):
        
        self.brand = brand
    
    def _get_interactions(self):

        bucket_uri = f'C:/Users/imbrigliaf/Documents/GitRepo/RecAll/example/data/'

        if self.brand == 'missoni':
        
          dataset = pd.read_csv(bucket_uri + f'{self.brand}.csv')


        elif self.brand == 'ton':
          
          clickstream = pd.read_csv(bucket_uri + f'{self.brand}.csv')
          metadata = pd.read_csv(bucket_uri + 'anagrafica_ton.csv')

          clickstream = (clickstream\
                         .groupby(['user_ids', 'product_code'])['brand'].count() 
                         ).reset_index()
          
          clickstream.columns = ['hashedEmail', 'product', 'actions']

          dataset = pd.merge(clickstream, metadata, left_on='product', right_on='pty_pim_variant')
          
          dataset = dataset[['hashedEmail', 'product', 'macro', 'saleline', 'actions']]
          dataset.columns = ['hashedEmail', 'product', 'macro', 'saleLine', 'actions']

          dataset['gender'] = 'W'

        return dataset 
    
    
    def _encondig_label(self, dataset, input_col, output_col):
        
        encoder = LabelEncoder()
        dataset[output_col] = encoder.fit(dataset[input_col]).transform(dataset[input_col])
        
        return dataset, encoder
    
    
    def fit(self, metadata=None, seasons=None):
        
        dataset = self._get_interactions()
        self.metadata = metadata
        
        if seasons:
            dataset = dataset[dataset['season'].isin(seasons)]
        
        ### Label Encoding
        dataset, _ = self._encondig_label(dataset, input_col='hashedEmail', output_col='user_id')
        dataset, _ = self._encondig_label(dataset, input_col='product', output_col='item_id')
        
        if metadata is not None:
            output_list_name = []
            
            for meta in metadata:
                output_name = meta + '_id'
                dataset, _ = self._encondig_label(dataset, input_col=meta, output_col=output_name)
                output_list_name.append(output_name)                
            
            dataset['metadata'] = dataset[output_list_name].values.tolist()
            self.metadata_id = output_list_name
            
        self.dataset = dataset
        
    def get_item_metadata_dict(self):
        
        if self.metadata is not None:
        
            return self.dataset.set_index('item_id')['metadata'].to_dict()
        
        else:
            
            return None
        

class CustomDataset(Dataset):

    """
    Class for loading data from a pandas DataFrame, containing user-product interactions/ratings/purchases
    plus item side information like product name, product category, etc.

    It returns a PyTorch Dataset object, which can be used to load data in a PyTorch model.

    In the __getitem__ method, the dataset is loaded in batches, and the data is returned as a dictionary.

    Also negative items are sampled from the same dataset, and returned as a dictionary.

    :param dataset: pandas.DataFrame
    :param user_id_col: str
    :param item_id_col: str
    :param metadata_id_col: list of str
    """
    
    def __init__(self, dataset, user_id_col, item_id_col, metadata_id_col):
        
        self.dataset = dataset
        self.user_id = user_id_col
        self.item_id = item_id_col
        self.metadata_id = metadata_id_col
        self.num_items = len(self.dataset[self.item_id].unique())

        self.dataset['negative_item_id'] = self._get_negative_items()
        self.dataset = self._apply_negative_metadata()
        self.negative_metadata_id = self._get_negative_metadata_column_names()

    def __len__(self) -> int:
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

        metadata.columns = ['negative_item_id'] + metadata_negative_names

        return metadata

    def _get_negative_metadata_column_names(self):

        return ['neg_'+metadata_column for metadata_column in self.metadata_id]

    def _apply_negative_metadata(self):

        metadata = self._get_negative_metadata()

        dataset = pd.merge(self.dataset, metadata, on='negative_item_id')

        return dataset

    def __getitem__(self, idx):
        
        return {'user_id': self.dataset.iloc[idx][self.user_id],
                'positive_item': {
                    'item_id': self.dataset.iloc[idx][self.item_id],
                    'metadata_id': {key: value for 
                                    key, value in zip(self.metadata_id, 
                                                      self.dataset.iloc[idx][self.metadata_id])}
                            },
                'negative_item': {
                    'item_id': self.dataset.iloc[idx]['negative_item_id'],
                    'metadata_id': {key: value for 
                                    key, value in zip(self.metadata_id, 
                                                      self.dataset.iloc[idx][self.negative_metadata_id])}
                    }
                }
              
               