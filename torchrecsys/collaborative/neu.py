# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from recall.embeddings.init_embeddings import ScaledEmbedding
from recall.helper.cuda import cpu, gpu

class NeuCF(torch.nn.Module):
    
    def __init__(self, dataset, n_factors, use_metadata=True):
        super().__init__(dataset, n_factors)
        
        self.use_metadata = use_metadata
        
        if use_metadata:
            self.n_metadata = self._get_n_metadata(dataset)
            self.n_metadata_type = self._get_n_metadata_type(dataset)
            self.metadata_gmf = ScaledEmbedding(self.n_metadata, n_factors)
            self.metadata_mlp = ScaledEmbedding(self.n_metadata, n_factors)
            
        else:
            self.n_metadata_type = 0
    
        self.user_gmf = ScaledEmbedding(self.n_users, self.n_factors)
        self.user_mlp = ScaledEmbedding(self.n_users, self.n_factors)
        
        self.item_gmf = ScaledEmbedding(self.n_items, self.n_factors)
        self.item_mlp = ScaledEmbedding(self.n_items, self.n_factors)
        
        self.linear_1 = torch.nn.Linear(n_factors*(2+self.n_metadata_type), self.n_factors*4)
        self.linear_2 = torch.nn.Linear(self.n_factors*4, self.n_factors*2)
        self.linear_3 = torch.nn.Linear(self.n_factors*2, self.n_factors)
        self.linear_4 = torch.nn.Linear(self.n_factors*2, 1)
        
        self.weights = torch.nn.Parameter(torch.rand(2), requires_grad=True)
        
    def _get_n_metadata(self, dataset):
        
        n_metadata = 0
        
        for col in dataset.metadata_id:
            n_metadata += dataset.dataset[col].max() + 1
        
        return n_metadata
    
    def _get_n_metadata_type(self, dataset):
        
        return len(dataset.metadata)
    
        
    def gmf(self, dataset, batch_size=1):
        
        """
        """
        
        user = Variable(torch.LongTensor(dataset['user_id'].values))
        item = Variable(torch.LongTensor(dataset['item_id'].values))
        
        if self.use_metadata:
            metadata = Variable(torch.LongTensor(list(dataset['metadata'])))
            metadata = self.metadata_gmf(metadata)
            
        user = self.user_gmf(user)
        item = self.item_gmf(item)
        
        if self.use_metadata:
            item = item.reshape(batch_size, 1, self.n_factors)        
            item_metadata = torch.cat([item, metadata], axis=1)

            ### sum of latent dimensions
            item = item_metadata.sum(1)
        
        #net = (user * item).sum(1).view(-1,1) 
        net = (user * item)
        
        return net
    
    def mlp(self, dataset, batch_size=1):
        
        """
        """
        user = Variable(torch.LongTensor(dataset['user_id'].values))
        item = Variable(torch.LongTensor(dataset['item_id'].values))
        
        if self.use_metadata:
            metadata = Variable(torch.LongTensor(list(dataset['metadata'])))
            metadata = self.metadata_mlp(metadata).reshape(batch_size, self.n_factors*self.n_metadata_type)
            
        user = self.user_mlp(user)
        item = self.item_mlp(item)
        
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
        user = Variable(torch.LongTensor(dataset['user_id'].values))
        item = Variable(torch.LongTensor(dataset['item_id'].values))
        
        gmf = self.gmf(dataset, batch_size)
        mlp = self.mlp(dataset, batch_size)
        
        net = torch.cat([gmf, mlp], axis=1)
        
        net = self.linear_4(net)
#        net = torch.nn.functional.sigmoid(net)
#         net = (self.weights * net).sum(1)
                
        return net