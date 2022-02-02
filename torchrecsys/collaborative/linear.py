# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding, ZeroEmbedding
from torchrecsys.helper.cuda import gpu
import pandas as pd
import numpy as np

class Linear(torch.nn.Module):
    
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
    
    
    def forward(self, batch, type):
        
        """
        Forward method that express the model as the dot product of user and item embeddings, plus the biases. 
        Item Embeddings itself is the sum of the embeddings of the item ID and its metadata
        """
        
        user = gpu(batch['user_id'], self.use_cuda)
        item = gpu(batch[type]['item_id'], self.use_cuda)
        
        ####
        if self.use_metadata:
            metadata = Variable(gpu(torch.LongTensor(list(batch['metadata'])), self.use_cuda))
            metadata = self.metadata(metadata)
        ###

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        
        user_embedding = self.user(user)
        item_embedding = self.item(item)
        
        ###
        if self.use_metadata:
        
            ### Reshaping in order to match metadata tensor
            item = item.reshape(len(batch['item_id'].values), 1, self.n_factors)        
            item_metadata = torch.cat([item, metadata], axis=1)

            ### sum of latent dimensions
            item = item_metadata.sum(1)
        ###

        net = (user_embedding * item_embedding).sum(1).view(-1,1) + user_bias + item_bias
        
        return net