# -*- coding: utf-8 -*-

import torch
import numpy as np
from recall.collaborative.lightfm import LightFM
from recall.collaborative.mlp import MLP
from recall.collaborative.ease import EASE
from recall.collaborative.neu import NeuCF
from recall.helper.cuda import gpu, cpu
from recall.helper.loss import hinge_loss
from recall.helper.evaluate import auc_score
from recall.helper.negative_sampling import get_negative_batch


class Recall(torch.nn.Module):
    
    """
    Encodes users (or item sequences) and items in a low dimensional space, using dot products as similarity measure 
    and Collaborative Filterings or Sequence Models as backend.
    
    ----------
    dataset: class
        instance of dataset class
    n_factors: int
        dimensionality of the embedding space
    net_type: string
        type of the model/net.
        "LightFM" -> Collaborative Filtering with (optional) item metadata with add operator and dot product
        "MLP" -> Collaborative Filtering with (optional) item metadata with concat operator and a stack of linear layers
        "NeuCF" -> Combine LightFM and MLP with a concat layer and a stack of linear layers 
        "EASE" -> Embarassingly Shallow Auto-Encoder
        "LSTM" -> Sequence Model using LSTM
    use_metadata: boolean
        Use True to add metadata to training procedure
    use_cuda: boolean, optional
        Use CUDA as backend. Default to False
    """
    
    def __init__(self, dataset, n_factors, net_type, use_metadata, use_cuda=False):
        super().__init__()
             
        self.dataset = dataset
        self.n_users = dataset.dataset['user_id'].max() + 1
        self.n_items = dataset.dataset['item_id'].max() + 1
        self.dictionary = dataset.get_item_metadata_dict()
        self.n_metadata = self._get_n_metadata(self.dataset)
        
        self.n_factors = n_factors
        
        self.use_cuda = use_cuda

        self.net_type = net_type
        self.use_metadata = use_metadata

        self._init_net(net_type=net_type)

    
    def _get_n_metadata(self, dataset):
        
        n_metadata = 0
        
        for col in dataset.metadata_id:
            n_metadata += dataset.dataset[col].max() + 1
        
        return n_metadata

    
    def _init_net(self, net_type='lightfm'):

        if net_type == 'lightfm':
          print('Training LightFM')
          self.net = LightFM(n_users=self.n_users, 
                              n_items=self.n_items, 
                              n_metadata=self.n_metadata, 
                              n_factors=self.n_factors, 
                              use_metadata=self.use_metadata, 
                              use_cuda=self.use_cuda)
        elif net_type == 'mlp':
          print('MLP under_construction')
          #net = MLP(n_users, n_items, n_metadata, n_metadata_type, n_factors, use_metadata=True, use_cuda=False)
          
        elif net_type == 'ease':
          print('EASE under construction')
          
        elif net_type == 'neucf':
          print('NeuCF under construction')
        

    def forward(self, net, batch, batch_size):

        score = gpu(net.forward(batch, batch_size), self.use_cuda)

        return score
    
    
    def backward(self, positive, negative, optimizer):
                
        optimizer.zero_grad()
                
        loss_value = hinge_loss(positive, negative)                

        loss_value.backward()
        
        optimizer.step()
        
        return loss_value.item()
    
    
    def fit(self, optimizer, batch_size=1024, epochs=10, split_train_test=False, verbose=False):
        
        if split_train_test:
            
            print('|== Splitting Train/Test ==|')
            
            data = self.dataset.dataset.iloc[np.random.permutation(len(self.dataset.dataset))]
            train = data.iloc[:int(len(data) * 0.9)]
            test = data.iloc[int(len(data) * 0.9):]
            
            print(f'Shape Train: {len(train)} \nShape Test: {len(test)}')
            
        else:
            
            train = self.dataset.dataset
        
        
        self.total_train_auc = []
        self.total_test_auc = []
        self.total_loss = []


        for epoch in range(epochs):

            self.net = self.net.train()

            print(f'Epoch: {epoch+1}')
            
            epoch_loss = []

            for first in range(0, len(train), batch_size):
                
                batch = train.iloc[first:first+batch_size, :]
                
                positive = self.forward(net=self.net, batch=batch, batch_size=batch_size)

                neg_batch = get_negative_batch(batch, self.n_items, self.dictionary)
                negative = self.forward(net=self.net, batch=neg_batch, batch_size=batch_size)     
                                                                
                loss_value = self.backward(positive, negative, optimizer)
                
            epoch_loss.append(loss_value)
            self.total_loss.append(epoch_loss)
            
            
            if verbose:
                ### AUC Calc.
                ### Train
                self.net = self.net.eval()
                
                train_sample = train.sample(n=20_000)

                positive_train = self.forward(net=self.net, batch=train_sample, batch_size=len(train_sample))

                neg_batch = get_negative_batch(train_sample, self.n_items, self.dictionary)  
                negative_train = self.forward(net=self.net, batch=neg_batch, batch_size=len(train_sample))           
                
                train_auc = auc_score(positive_train, negative_train)
                self.total_train_auc.append(train_auc)
                
                ### Test
                test_sample = test.sample(n=20_000) 
                
                positive_test = self.forward(net=self.net, batch=test_sample, batch_size=len(test_sample))

                neg_batch = get_negative_batch(test_sample, self.n_items, self.dictionary)
                negative_test = self.forward(net=self.net, batch=neg_batch, batch_size=len(test_sample))   

                test_auc = auc_score(positive_test, negative_test)
                self.total_test_auc.append(test_auc)
                
                print(f'== Loss: {sum(epoch_loss)} \n== Train AUC: {train_auc} \n== Test AUC: {test_auc}')
            
    def history(self):
        
        return {'train_loss': self.total_loss,
                'train_auc': self.total_train_auc,
                'test_auc': self.total_test_auc}
    
    def get_item_representation(self):
        
        return self.item.weight.cpu().detach().numpy()