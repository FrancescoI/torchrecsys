# -*- coding: utf-8 -*-

from typing import List
import torch
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding, ZeroEmbedding
from torchrecsys.helper.cuda import gpu

class Linear(torch.nn.Module):

    """
    It trains a linear collaborative filtering using content data (eg: product category, brand, etc) as optional.
    The user x item matrix is factorized into two matrices: user and item embeddings, of shape (n_users, n_factors) and (n_items, n_factors).

    Parameters:
    -----------
    n_users: int -> number of unique users in the dataset
    n_items: int -> number of unique items in the dataset
    n_metadata: int -> number of unique metadata in the dataset
    n_factors: int -> number of latent factors to be used in the factorization
    use_metadata: bool -> whether to use metadata or not
    use_cuda: bool -> whether to use cuda or not
    """
    
    def __init__(self,
                 n_users, 
                 n_items, 
                 n_metadata, 
                 n_factors, 
                 use_metadata=True, 
                 use_cuda=False):
        
        super(Linear, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda
        
        if use_metadata:
            self.metadata = torch.nn.ModuleList(
                                [ScaledEmbedding(size, self.n_factors, sparse=True) for _ , size in self.n_metadata.items()])
        
        self.user = ScaledEmbedding(self.n_users, self.n_factors, sparse=True)
        self.item = ScaledEmbedding(self.n_items, self.n_factors, sparse=True)
        
        self.user_bias = ZeroEmbedding(self.n_users, 1, sparse=True)
        self.item_bias = ZeroEmbedding(self.n_items, 1, sparse=True)
    
    
    def forward(self, batch, user_key, item_key, metadata_key=None):
        
        """
        Forward method that express the model as the dot product of user and item embeddings, plus the biases. 
        Item Embeddings itself is the sum of the embeddings of the item ID and its metadata
        """
        
        user = batch[user_key]
        item = batch[item_key]
        
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        
        user_embedding = self.user(user)
        item_embedding = self.item(item)

        #### metadata
        if self.use_metadata:
            metadata = batch[metadata_key]
            
            for idx, layer in enumerate(self.metadata):
                item_embedding += layer(metadata[:, idx])
        ###

        net = (user_embedding * item_embedding).sum(1).view(-1,1) + user_bias + item_bias
        
        return net