# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MyDataset():
    
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
        
        
class Dataset():
    
    def __init__(self, dataset):
        
        pass
        
        
    
    
    
    
    
    
    
    