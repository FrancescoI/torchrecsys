# -*- coding: utf-8 -*-

import torch
import json
from torchrecsys.dataset.dataset import FastDataLoader
from torchrecsys.collaborative.linear import Linear
from torchrecsys.collaborative.mlp import MLP
from torchrecsys.collaborative.fm import FM
from torchrecsys.helper.cuda import gpu
from torchrecsys.helper.loss import hinge_loss
from torchrecsys.helper.evaluate import auc_score
from torchrecsys.evaluate.metrics import Metrics
import pandas as pd
from typing import List


class TorchRecSys(torch.nn.Module):
    
    """
    Encodes users (or item sequences) and items in a low dimensional space, using dot products as similarity measure 
    and Collaborative Filterings or Sequence Models as backend.
    
    ----------
    dataloader: class
        instance of pytorch dataloader class
    n_factors: int
        dimensionality of the embedding space
    net_type: string
        type of the model/net.
        "linear" -> CF with (optional) side information about item (add operator) 
        "mlp" -> CF with (optional) side information about item (concat operator) and a stack of linear layers
        "neucf" -> Combine linear and mlp with a concat layer and a linear layer 
        "ease" -> Embarassingly Shallow Auto-Encoder
        "lstm" -> Sequence Model using LSTM
    use_metadata: boolean
        Use True to add metadata to training procedure
    use_cuda: boolean, optional
        Use CUDA as backend. Default to False
    """
    
    def __init__(self,
                 path: str, 
                 n_factors: int = 80, 
                 net_type: str = 'linear', 
                 metadata_name: List[str] = None, 
                 use_cuda: bool = False,
                 debug: bool = False):

        super().__init__()

        self.path = path     
        self.config = self._read_metadata(self.path)
        self.n_users = self.config.get('num_users')
        self.n_items = self.config.get('num_items')      
        self.metadata_size = self.config.get('num_metadata')

        self.n_factors = n_factors
        
        self.use_cuda = use_cuda

        self.net_type = net_type
        
        if metadata_name:
            self.metadata_name = metadata_name
            self.use_metadata = True
        else:
            self.use_metadata = False

        self.debug = debug

        self._init_net(net_type=net_type)

    def _read_metadata(self, path):

        with open(f'{path}/config.json') as json_file:    
            config = json.load(json_file)

        return config

    def _init_net(self, net_type='linear'):

        assert net_type in ('linear', 'mlp', 'neucf', 'fm', 'lstm'), 'Net type must be one of "linear", "mlp", "neu", "ease" or "lstm"'

        if net_type == 'linear':

          print('Linear Collaborative Filtering')

          self.net = Linear(n_users=self.n_users, 
                            n_items=self.n_items, 
                            n_metadata=self.metadata_size, 
                            n_factors=self.n_factors, 
                            use_metadata=self.use_metadata, 
                            use_cuda=self.use_cuda)
        
        elif net_type == 'mlp':

            print('Multi Layer Perceptron')

            self.net = MLP(n_users=self.n_users, 
                           n_items=self.n_items, 
                           n_metadata=self.metadata_size, 
                           n_factors=self.n_factors, 
                           use_metadata=self.use_metadata, 
                           use_cuda=self.use_cuda)
          
        elif net_type == 'fm':

            print('Factorization Machine')

            self.net = FM(n_users=self.n_users, 
                          n_items=self.n_items, 
                          n_metadata=self.metadata_size, 
                          n_factors=self.n_factors, 
                          use_metadata=self.use_metadata, 
                          use_cuda=self.use_cuda)
          
        elif net_type == 'neucf':
            NotImplementedError('NeuCF not implemented yet')

        elif net_type == 'lstm':
            NotImplementedError('LSTM not implemented yet')

        self.net = gpu(self.net, self.use_cuda)


    def forward(self, net, batch):

        positive_score = gpu(net.forward(batch, 
                                         user_key='user_id', 
                                         item_key='pos_item_id',
                                         metadata_key='pos_metadata_id'),
                             self.use_cuda)

        negative_score = gpu(net.forward(batch, 
                                         user_key='user_id', 
                                         item_key='neg_item_id',
                                         metadata_key='neg_metadata_id'),
                             self.use_cuda)

        return positive_score, negative_score
    
    
    def backward(self, positive, negative, optimizer):
                
        optimizer.zero_grad()
                
        loss_value = hinge_loss(positive, negative)                

        loss_value.backward()
        
        optimizer.step()

        return loss_value.item()
    
      
    def fit(self, optimizer, epochs=10, batch_size=512):
        
        for _ in range(epochs):

            self.net = self.net.train()
            
            train_loader = FastDataLoader(path=f'{self.path}/train.csv',
                                          metadata_name=self.metadata_name,
                                          batch_size=batch_size)
            
            for batch in train_loader:

                mini_batch = {}

                for key, values in batch.items():
                    mini_batch.update({key: gpu(values, self.use_cuda)})
                    
                positive, negative = self.forward(net=self.net, batch=mini_batch)
                loss_value = self.backward(positive, negative, optimizer)

            print(f'|--- Training Loss: {loss_value}')
            print(f'|--- Training AUC: {auc_score(positive, negative)}')


    def evaluate(self, batch_size=512, metrics=['loss', 'auc']):

        self.net = self.net.eval()

        measures = Metrics()

        metrics = {'loss': [],
                   'auc': [],
                   'hit_rate': []
                  }
        loss = []
        auc = []
        hit_rate = []

        test_loader = FastDataLoader(path=f'{self.path}/test.csv',
                                     metadata_name=self.metadata_name,
                                     batch_size=batch_size)

        for batch in test_loader:

            mini_batch = {}

            for key, values in batch.items():
                mini_batch.update({key: gpu(values, self.use_cuda)})

            positive_test, negative_test = self.forward(net=self.net, batch=mini_batch)
            
            if 'auc' in metrics:
                auc_score = measures.auc_score(positive_test, negative_test)
                auc.append(auc_score)
                
            if 'loss' in metrics: 
                loss_value_test = hinge_loss(positive_test, negative_test)
                loss.append(loss_value_test)

        metrics.update({'auc': sum(auc) / len(auc),
                        'loss': sum(loss) / len(loss)})

        for metric, value in metrics.items():
            print(f'|--- Testing {metric}: {value}')


    def predict(self, user_id: int, top_k: int = 10):
        
        """
        It returns sorted item indexes for a given user or a list of users.
        """

        self.net = self.net.eval()

        batch = gpu(self._create_inference_batch(user_id), self.use_cuda)

        score = self.net.forward(batch,
                                 user_key='user_id',
                                 item_key='pos_item_id',
                                 metadata_key='pos_metadata_id')

        sorted_index = torch.argsort(score, dim=0, descending=True)

        if top_k:
            sorted_index = sorted_index[:top_k]

        return sorted_index

    
    def _create_inference_batch(self, user_id):

        dataframe = pd.DataFrame({'user_id': [user_id] * self.n_items,
                                  'pos_item_id': list(range(self.n_items)),
                                })

        if self.metadata_name:
            metadata = self._get_metadata()
            dataframe = pd.merge(dataframe, metadata, on='pos_item_id', how='inner')
            dataframe['pos_metadata_id'] = dataframe[self.metadata_name].values.tolist()

        batch = {
                'user_id': torch.from_numpy(dataframe['user_id'].values),
                'pos_item_id': torch.from_numpy(dataframe['pos_item_id'].values)
                }

        if self.metadata_name:
            batch.update({'pos_metadata_id': torch.Tensor(dataframe['pos_metadata_id']).long()})

        return batch

    def _get_metadata(self):
        
        metadata = pd.read_csv(f'{self.path}/meta.csv')
            
        return metadata