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

    def predict(self, user_ids, top_k=None):

        """
        It returns sorted item indexes for a given user.
        """

        num_users = len(user_ids)

        user_embedding = self.user(torch.tensor(user_ids)).reshape(num_users, 1, self.n_factors).repeat(1, self.n_items, 1)
        item_embedding = self.item.weight.data

        if self.use_metadata:
            mapping = gpu(torch.from_numpy(self.dataloader.map_item_metadata()), self.use_cuda)
            
            for idx, layer in enumerate(self.metadata):
                single_layer = layer.weight.data
                item_meta_embedding = torch.cat([item_meta_embedding, single_layer], axis=0) if idx > 0 else single_layer

            item_meta_embedding = torch.cat([item_meta_embedding, item_embedding], axis=0)

            mapping = mapping.reshape(mapping.shape[0], 1, mapping.shape[1]) # (n_items X (n_items + n_metadata))
            item_meta_embedding = item_meta_embedding.transpose(1, 0).reshape(1, self.n_factors, mapping.shape[2]) # (1 X n_factors X (n_items + n_metadata))

            cat = mapping * item_meta_embedding
            item_embedding = cat[cat != 0].reshape(self.n_items, self.n_factors, 1+self.n_distinct_metadata)
            item_embedding = item_embedding.transpose(2,1)
            item_embedding = item_embedding.reshape(self.n_items, self.n_factors * (1+self.n_distinct_metadata))
        ###

        item_embedding = item_embedding.reshape(1, self.n_items, self.n_factors * (1+self.n_distinct_metadata)).repeat(num_users, 1, 1)
        cat = torch.cat([user_embedding, item_embedding], axis=2)

        net = self.linear_1(cat)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_2(net)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_3(net)

        sorted_index = torch.argsort(net, dim=0, descending=True).squeeze()

        if top_k:
            sorted_index = sorted_index[:, :top_k].squeeze()

        return sorted_index