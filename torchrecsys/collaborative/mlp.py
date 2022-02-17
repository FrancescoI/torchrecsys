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
        
        self.input_shape = self.n_factors * 2

        if use_metadata:
            n_metadata = len(self.n_metadata.keys())
            self.input_shape += (self.n_factors * n_metadata)

        self.user = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)
        
        self.linear_1 = gpu(torch.nn.Linear(self.input_shape, 1_024), self.use_cuda)
        self.linear_2 = gpu(torch.nn.Linear(1_024, 128), self.use_cuda)
        self.linear_3 = gpu(torch.nn.Linear(128, 1), self.use_cuda)

        if use_metadata:
            self.metadata = torch.nn.ModuleList(
                                gpu([ScaledEmbedding(size, self.n_factors) for _ , size in self.n_metadata.items()], 
                                self.use_cuda))


    def forward(self, batch, user_key, item_key, metadata_key=None):

        user = batch[user_key]
        item = batch[item_key]

        user_embedding = self.user(user)
        item_embedding = self.item(item)
        
        #### metadata
        if self.use_metadata:
            metadata = batch[metadata_key]
            
            for idx, layer in enumerate(self.metadata):
                single_layer = layer(metadata[:, idx])
                item_embedding = torch.cat([item_embedding, single_layer], axis=1)
        ###

        cat = torch.cat([user_embedding, item_embedding], axis=1)

        net = self.linear_1(cat)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_2(net)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_3(net)
        
        return net

    def predict(self, user_id, top_k=None):

        """
        It returns sorted item indexes for a given user.
        """

        user_emb = self.user(torch.tensor(user_id)).repeat(self.n_items, 1) 
        item_emb = self.item.weight.data

        cat = torch.cat([user_emb, item_emb], axis=1)

        net = self.linear_1(cat)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_2(net)
        net = torch.nn.functional.relu(net)
        
        prediction = self.linear_3(net)

        sorted_index = torch.argsort(prediction, dim=0, descending=True)

        if top_k:
            sorted_index = sorted_index[:top_k].squeeze()

        return sorted_index