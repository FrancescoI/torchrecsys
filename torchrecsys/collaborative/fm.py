from multiprocessing.sharedctypes import Value
import torch
from torch.autograd import Variable
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding, ZeroEmbedding
from torchrecsys.helper.cuda import gpu
import pandas as pd
import numpy as np

class FM(torch.nn.Module):

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
        super(FM, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda

        self.n_input = self.n_users + self.n_items
        
        self.user = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)
                
        self.linear_user = gpu(ScaledEmbedding(self.n_users, 1), self.use_cuda)
        self.linear_item = gpu(ScaledEmbedding(self.n_items, 1), self.use_cuda)

        if use_metadata:
            self.metadata = torch.nn.ModuleList(
                                [gpu(ScaledEmbedding(size, self.n_factors), self.use_cuda) 
                                 for _ , size in self.n_metadata.items()])

            self.linear_metadata = torch.nn.ModuleList(
                                [gpu(ScaledEmbedding(size, 1), self.use_cuda) 
                                 for _ , size in self.n_metadata.items()])



    def forward(self, batch, user_key, item_key, metadata_key=None):
        
        """
        Forward method that express the model as the dot product of user and item embeddings, plus the biases. 
        Item Embeddings itself is the sum of the embeddings of the item ID and its metadata
        """
        
        user = batch[user_key]
        item = batch[item_key]
        
        user_embedding = self.user(user).reshape(user.shape[0], 1, self.n_factors)
        item_embedding = self.item(item).reshape(item.shape[0], 1, self.n_factors)

        #### metadata
        if self.use_metadata:
            metadata = batch[metadata_key]
            
            for idx, layer in enumerate(self.metadata):
                layer = layer(metadata[:, idx]).reshape(item.shape[0], 1, self.n_factors)
                item_embedding = torch.cat([item_embedding, layer], dim=1) 

        embedding = torch.cat([user_embedding, item_embedding], dim=1)

        power_of_sum = embedding.sum(dim=1).pow(2)
        sum_of_power = embedding.pow(2).sum(dim=1)

        pairwise = (power_of_sum - sum_of_power).sum(1) * 0.5

        ### Linear
        user_linear = self.linear_user(user).reshape(user.shape[0], 1, 1)
        item_linear = self.linear_item(item).reshape(item.shape[0], 1, 1)

        if self.use_metadata:
            for idx, layer in enumerate(self.linear_metadata):
                lin_layer = layer(metadata[:, idx]).reshape(item.shape[0], 1, 1)
                item_linear = torch.cat([item_linear, lin_layer], dim=1)

        linear = torch.cat([user_linear, item_linear], dim=1).sum(1).reshape(user.shape[0],)

        net = torch.sigmoid(linear + pairwise)
        
        return net


    def predict(self, user_id, top_k=None):

        """
        It returns sorted item indexes for a given user.
        """
        
        # LINEAR
        user_linear = self.linear_user(torch.tensor(user_id)).repeat(self.n_items, 1) 
        item_linear = self.linear_item.weight.data
        
        linear = torch.cat([user_linear, item_linear], dim=1).sum(1)

        # EMBEDDING
        user_embedding = self.user(torch.tensor(user_id)).repeat(self.n_items, 1).reshape(self.n_items, 1, self.n_factors) 
        item_embedding = self.item.weight.data.reshape(self.n_items, 1, self.n_factors) 

        embedding = torch.cat([user_embedding, item_embedding], dim=1)

        power_of_sum = embedding.sum(dim=1).pow(2)
        sum_of_power = embedding.pow(2).sum(dim=1)

        pairwise = (power_of_sum - sum_of_power).sum(1) * 0.5

        ### COMBINE
        prediction = torch.sigmoid(linear + pairwise)

        sorted_index = torch.argsort(prediction, dim=0, descending=True)

        if top_k:
            sorted_index = sorted_index[:top_k].squeeze()

        return sorted_index