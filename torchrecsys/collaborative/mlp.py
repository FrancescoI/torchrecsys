# -*- coding: utf-8 -*-

import torch
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding
from torchrecsys.helper.cuda import gpu

class MLP(torch.nn.Module):
    
    def __init__(self, 
                 dataloader,
                 n_users, 
                 n_items, 
                 n_metadata, 
                 n_factors, 
                 use_metadata=True, 
                 use_cuda=False):
                 
        super(MLP, self).__init__()

        self.dataloader = dataloader

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda
        
        self.input_shape = self.n_factors * 2

        if use_metadata:
            self.n_distinct_metadata = len(self.n_metadata.keys())
            self.input_shape += (self.n_factors * self.n_distinct_metadata)

        self.user = ScaledEmbedding(self.n_users, self.n_factors)
        self.item = ScaledEmbedding(self.n_items, self.n_factors)
        
        self.linear_1 = torch.nn.Linear(self.input_shape, 1_024)
        self.linear_2 = torch.nn.Linear(1_024, 128)
        self.linear_3 = torch.nn.Linear(128, 1)

        if use_metadata:
            self.metadata = torch.nn.ModuleList(
                                [ScaledEmbedding(size, self.n_factors) for _ , size in self.n_metadata.items()])

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