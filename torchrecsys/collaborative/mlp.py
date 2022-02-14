# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding, ZeroEmbedding
from torchrecsys.helper.cuda import gpu
import pandas as pd
import numpy as np


class MLP(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_metadata, n_factors, use_metadata=True, use_cuda=False):
        super(MLP, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda
        
        if use_metadata:
            self.n_metadata = self.n_metadata
            self.metadata = gpu(ScaledEmbedding(self.n_metadata, n_factors), self.use_cuda)

        self.input_shape = self.n_factors * 2

        self.user = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)
        
        self.user_bias = gpu(ZeroEmbedding(self.n_users, 1), self.use_cuda)
        self.item_bias = gpu(ZeroEmbedding(self.n_items, 1), self.use_cuda)
        
        self.linear_1 = gpu(torch.nn.Linear(self.input_shape, 1_024, self.use_cuda))
        self.linear_2 = gpu(torch.nn.Linear(1_024, 128, self.use_cuda))
        self.linear_3 = gpu(torch.nn.Linear(128, 1, self.use_cuda))
        
    def _get_n_metadata(self, dataset):
        
        n_metadata = 0
        
        for col in dataset.metadata_id:
            n_metadata += dataset.dataset[col].max() + 1
        
        return n_metadata
    
    def _get_n_metadata_type(self, dataset):
        
        return len(dataset.metadata)


    def forward(self, batch, user_key, item_key, metadata_key=None):

        user = batch[user_key]
        item = batch[item_key]

        user = self.user(user)
        item = self.item(item)
        
        if self.use_metadata:
            metadata = batch[metadata_key]
            metadata = self.metadata(metadata)
                
            ### Reshaping in order to match metadata tensor
            item = item.reshape(len(batch['item_id'].values), 1, self.n_factors)

            item_metadata = torch.cat([item, metadata], axis=1)
            item = item_metadata.sum(1)

        cat = torch.cat([user, item], axis=1)

        net = self.linear_1(cat)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_2(net)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_3(net)
        
        return net