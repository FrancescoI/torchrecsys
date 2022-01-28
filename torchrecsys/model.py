# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchrecsys.collaborative.linear import Linear
from torchrecsys.collaborative.mlp import MLP
from torchrecsys.collaborative.ease import EASE
from torchrecsys.collaborative.neu import NeuCF
from torchrecsys.helper.cuda import gpu, cpu
from torchrecsys.helper.loss import hinge_loss
from torchrecsys.helper.evaluate import auc_score
from torchrecsys.helper.negative_sampling import get_negative_batch


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
                 use_cuda: bool = False):

        super().__init__()
             
        self.dataloader = dataloader

        self.n_users = dataloader.dataset.num_users
        self.n_items = dataloader.dataset.num_items      
        self.n_metadata = len(dataloader.dataset.metadata_id)

        self.n_factors = n_factors
        
        self.use_cuda = use_cuda

        self.net_type = net_type
        self.use_metadata = use_metadata

        self._init_net(net_type=net_type)

    
    def _init_net(self, net_type='linear'):

        assert (net_type in ('linear', 'mlp', 'neu', 'ease', 'lstm'), 
                'Net type must be one of "linear", "mlp", "neu", "ease" or "lstm"')

        if net_type == 'linear':

          print('Training Linear Collaborative Filtering')

          self.net = Linear(n_users=self.n_users, 
                            n_items=self.n_items, 
                            n_metadata=self.n_metadata, 
                            n_factors=self.n_factors, 
                            use_metadata=self.use_metadata, 
                            use_cuda=self.use_cuda)
        
        elif net_type == 'mlp':
            NotImplementedError('MLP not implemented yet')
          
        elif net_type == 'ease':
            NotImplementedError('EASE not implemented yet')
          
        elif net_type == 'neucf':
            NotImplementedError('NeuCF not implemented yet')

        elif net_type == 'lstm':
            NotImplementedError('LSTM not implemented yet')


    def forward(self, net, batch, type):

        score = gpu(net.forward(batch, type), self.use_cuda)

        return score
    
    
    def backward(self, positive, negative, optimizer):
                
        optimizer.zero_grad()
                
        loss_value = hinge_loss(positive, negative)                

        loss_value.backward()
        
        optimizer.step()
        
      
    def fit(self, dataloader, optimizer, epochs=10):
        
        for epoch in range(epochs):

            self.net = self.net.train()

            print(f'Epoch: {epoch+1}')
            
            for batch in dataloader:
                                
                positive = self.forward(net=self.net, batch=batch, type='positive_item_id')
                negative = self.forward(net=self.net, batch=batch, type='negative_item_id')

                loss_value = self.backward(positive, negative, optimizer)