# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from recall.embeddings.init_embeddings import ScaledEmbedding
from recall.helper.cuda import gpu
import pandas as pd
import numpy as np


class MLP(torch.nn.Module):

    ### UNDER COSTRUCTION
    
    def __init__(self, n_users, n_items, n_metadata, n_metadata_type, n_factors, use_metadata=True, use_cuda=False):
        super(MLP, self).__init__()
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda

        if use_metadata:
            self.n_metadata = n_metadata
            self.n_metadata_type = n_metadata_type
            self.metadata = gpu(ScaledEmbedding(self.n_metadata, n_factors), self.use_cuda)
            
        else:
            self.n_metadata_type = 0
        
        self.linear_1 = gpu(torch.nn.Linear(n_factors*(2+self.n_metadata_type), int(self.n_factors/2)), self.use_cuda)
        self.linear_2 = gpu(torch.nn.Linear(int(self.n_factors/2), int(self.n_factors/4)), self.use_cuda)
        self.linear_3 = gpu(torch.nn.Linear(int(self.n_factors/4), 1), self.use_cuda)
        
    def _get_n_metadata(self, dataset):
        
        n_metadata = 0
        
        for col in dataset.metadata_id:
            n_metadata += dataset.dataset[col].max() + 1
        
        return n_metadata
    
    def _get_n_metadata_type(self, dataset):
        
        return len(dataset.metadata)
    
    
    def mlp(self, dataset, batch_size=1):
        
        """
        """
        user = gpu(torch.from_numpy(dataset['user_id'].values), self.use_cuda)
        item = gpu(torch.from_numpy(dataset['item_id'].values), self.use_cuda)
        
        if self.use_metadata:
            metadata = Variable(gpu(torch.LongTensor(list(dataset['metadata'])), self.use_cuda))
            metadata = self.metadata(metadata).reshape(batch_size, self.n_factors*self.n_metadata_type)
            
        user = self.user(user)
        item = self.item(item)
        
        if self.use_metadata:
            cat = torch.cat([user, item, metadata], axis=1).reshape(batch_size, (2+self.n_metadata_type)*self.n_factors)
        else:
            cat = torch.cat([user, item], axis=1).reshape(batch_size, 2*self.n_factors)
                
        net = self.linear_1(cat)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_2(net)
        net = torch.nn.functional.relu(net)
        
        net = self.linear_3(net)
        
        return net
    
    def forward(self, dataset, batch_size=1):
        
        """
        """
        
        net = gpu(self.mlp(dataset, batch_size), self.use_cuda)
                
        return net
    
    def get_item_representation(self):
        
        if self.use_metadata:
            
            data = (self.dataset
                    .dataset[['item_id'] + self.dataset.metadata_id]
                    .drop_duplicates())
            
            mapping = pd.get_dummies(data, columns=[*self.dataset.metadata_id]).values[:, 1:]
            identity = np.identity(self.dataset.dataset['item_id'].max() + 1)
            binary = np.hstack([identity, mapping])
            
            metadata_representation = np.vstack([self.item.weight.detach().numpy(), self.metadata.weight.detach().numpy()])
            
            return np.dot(binary, metadata_representation), binary, metadata_representation
        
        else:
            return self.item.weight.cpu().detach().numpy()