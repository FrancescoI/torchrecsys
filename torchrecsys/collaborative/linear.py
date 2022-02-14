# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding, ZeroEmbedding
from torchrecsys.helper.cuda import gpu
import pandas as pd
import numpy as np

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
    
    def __init__(self, n_users, n_items, n_metadata, n_factors, use_metadata=True, use_cuda=False):
        super(Linear, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda
        
        if use_metadata:
            self.n_metadata = self.n_metadata
            self.metadata = gpu(ScaledEmbedding(self.n_metadata, n_factors), self.use_cuda)

        
        self.user = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)
        
        self.user_bias = gpu(ZeroEmbedding(self.n_users, 1), self.use_cuda)
        self.item_bias = gpu(ZeroEmbedding(self.n_items, 1), self.use_cuda)
    
    
    def forward(self, batch, user_key, item_key, metadata_key=None):
        
        """
        Forward method that express the model as the dot product of user and item embeddings, plus the biases. 
        Item Embeddings itself is the sum of the embeddings of the item ID and its metadata
        """
        
        user = batch[user_key]
        item = batch[item_key]
        
        ####
        if self.use_metadata:
            metadata = batch[metadata_key]
            metadata_embedding = self.metadata(metadata)
        ###

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        
        user_embedding = self.user(user)
        item_embedding = self.item(item)
        
        ###
        if self.use_metadata:
        
            ### Reshaping in order to match metadata tensor
            item_embedding = item_embedding.reshape(len(batch['item_id'].values), 1, self.n_factors)        
            item_metadata_embedding = torch.cat([item_embedding, metadata_embedding], axis=1)

            ### sum of latent dimensions
            item_embedding = item_metadata_embedding.sum(1)
        ###

        net = (user_embedding * item_embedding).sum(1).view(-1,1) + user_bias + item_bias
        
        return net

    def predict(self, user_id, top_k=None):

        """
        It returns sorted item indexes for a given user.
        """

        user_emb = self.user(torch.tensor(user_id)).repeat(self.n_items, 1) 
        item_emb = self.item.weight.data
        item_bias = self.item_bias.weight.data

        prediction = (user_emb * item_emb).sum(1).view(-1,1)
        prediction += item_bias

        sorted_index = torch.argsort(prediction, dim=0, descending=True)

        if top_k:
            sorted_index = sorted_index[:top_k].squeeze()

        return sorted_index