# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchrecsys.collaborative.linear import Linear
from torchrecsys.collaborative.mlp import MLP
from torchrecsys.collaborative.ease import EASE
from torchrecsys.collaborative.neu import NeuCF
from torchrecsys.collaborative.fm import FM
from torchrecsys.helper.cuda import gpu, cpu
from torchrecsys.helper.loss import hinge_loss
from torchrecsys.helper.evaluate import auc_score
from torchrecsys.evaluate.metrics import Metrics
import random
import pandas as pd


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
                 dataloader, 
                 n_factors: int = 80, 
                 net_type: str = 'linear', 
                 use_metadata: bool = False, 
                 use_cuda: bool = False,
                 debug: bool = False):

        super().__init__()
             
        self.dataloader = dataloader

        self.n_users = dataloader.num_users
        self.n_items = dataloader.num_items      
        self.metadata_size = dataloader.metadata_size

        self.n_factors = n_factors
        
        self.use_cuda = use_cuda

        self.net_type = net_type
        self.use_metadata = use_metadata

        self.debug = debug

        self._init_net(net_type=net_type)

        self.training = {}
        for key, values in dataloader.train.items():
            self.training.update({key: gpu(values, self.use_cuda)})

    
    def _init_net(self, net_type='linear'):

        assert net_type in ('linear', 'mlp', 'neucf', 'fm', 'lstm'), 'Net type must be one of "linear", "mlp", "neu", "ease" or "lstm"'

        if net_type == 'linear':

          print('Linear Collaborative Filtering')

          self.net = Linear(dataloader=self.dataloader,
                            n_users=self.n_users, 
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
        
        print('|-- Training model')
        for epoch in range(epochs):

            self.net = self.net.train()

            print(f'Epoch: {epoch+1}')
            
            for first in range(0, len(self.training['user_id']), batch_size):

                if self.debug:    
                    print(f'On total of {first / len(self.training["user_id"]) * 100:.2f}%')
                    
                batch = {k: v[first:first+batch_size] for k, v in self.training.items()}

                positive, negative = self.forward(net=self.net, batch=batch)

                loss_value = self.backward(positive, negative, optimizer)

            print(f'|--- Training Loss: {loss_value}')
            print(f'|--- Training AUC: {auc_score(positive, negative)}')


    def evaluate(self, metrics=['loss', 'auc']):

        self.net = self.net.eval()

        measures = Metrics()
        
        all_indices = range(len(self.dataloader.test['user_id']))
        random_indices = random.sample(all_indices, min(50_000, len(all_indices))) ### takes random 50k samples
        
        testing = {}
        for key, values in self.dataloader.test.items():
            testing.update({key: gpu(values[random_indices], self.use_cuda)})

        positive_test, negative_test = self.forward(net=self.net, batch=testing)

        if 'loss' in metrics: 
            loss_value_test = hinge_loss(positive_test, negative_test)
            print(f'|--- Testing Loss: {loss_value_test}')

        if 'auc' in metrics:
            auc_score = measures.auc_score(positive_test, negative_test)
            print(f'|--- Testing AUC: {auc_score}')

        if 'hit_rate' in metrics:

            testing_dictionary = (pd.DataFrame({'user_id': self.dataloader.test['user_id'][random_indices],
                                                'item_id': self.dataloader.test['pos_item_id'][random_indices]})
                                .groupby('user_id')['item_id']
                                .apply(list)
                                .to_dict())

            user_ids = list(testing_dictionary.keys())
            y_pred = self.net.predict(user_ids=user_ids, top_k=10)
            
            ### TO WRAP IN A FUNCTION
            y_hat_list = list(testing_dictionary.values())

            max_row_length = 1
            for row in y_hat_list:
                if len(row) > max_row_length:
                    max_row_length = len(row)

            y_hat_balanced_list = []
            for row in y_hat_list:
                if len(row) < max_row_length:
                    to_pad = max_row_length - len(row)
                    padding = row + [-1] * to_pad
                    y_hat_balanced_list.append(padding)
                else:
                    y_hat_balanced_list.append(row)

            y_hat = torch.from_numpy(np.array(y_hat_balanced_list))

            ###

            hit_rates = measures.hit_rate(y_hat=y_hat, y_pred=y_pred)

            print(f'|--- Testing Hit Rate: {hit_rates}')

    
    def predict(self, user_ids, top_k):
        
        self.net = self.net.eval()

        return self.net.predict(user_ids, top_k)