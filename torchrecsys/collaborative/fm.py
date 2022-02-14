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
        
        if not use_metadata:
            self.n_metadata = self.n_metadata   
            self.metadata = gpu(ScaledEmbedding(self.n_metadata, n_factors), self.use_cuda)
            self.n_input += self.n_metadata

        self.user = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)
                
        self.w0 = gpu(torch.nn.Parameter(torch.zeros(1)), self.use_cuda)
        self.bias = gpu(ScaledEmbedding(self.n_input, 1), self.use_cuda)


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
        
        user_embedding = self.user(user).reshape(user.shape[0], 1, self.n_factors)
        item_embedding = self.item(item).reshape(item.shape[0], 1, self.n_factors)

        embedding = torch.cat([user_embedding, item_embedding], dim=1)
        
        ###
        if self.use_metadata:
        
            ### Reshaping in order to match metadata tensor
            item_embedding = item_embedding.reshape(len(batch['item_id'].values), 1, self.n_factors)        
            item_metadata_embedding = torch.cat([item_embedding, metadata_embedding], axis=1)

            ### sum of latent dimensions
            item_embedding = item_metadata_embedding.sum(1)
        ###

        power_of_sum = embedding.sum(dim=1).pow(2)
        sum_of_power = embedding.pow(2).sum(dim=1)

        pairwise = (power_of_sum - sum_of_power).sum(1) * 0.5
        
        bias_u = self.bias(user)
        bias_i = self.bias(item)

        bias = (bias_u + bias_i).sum(1)

        net = torch.sigmoid(self.w0 + bias + pairwise)
        
        return net