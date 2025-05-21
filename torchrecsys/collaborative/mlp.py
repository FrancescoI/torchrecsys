# -*- coding: utf-8 -*-

import torch
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding
from torchrecsys.helper.cuda import gpu

class MLP(torch.nn.Module):
    
    def __init__(self, 
                 n_users, 
                 n_items, 
                 n_metadata, 
                 n_factors, 
                 use_metadata=True,
                 use_batch_norm: bool = True,
                 hidden_layers: List[int] = None, 
                 use_cuda=False):
        """
        Multi-Layer Perceptron (MLP) model for collaborative filtering.

        Concatenates user, item, and optional metadata embeddings, then passes them
        through a series of fully connected layers.

        Parameters:
        -----------
        n_users: int
            Number of unique users.
        n_items: int
            Number of unique items.
        n_metadata: dict
            Dictionary where keys are metadata feature names and values are the number
            of unique categories for that feature.
        n_factors: int
            Dimensionality of the embedding space for users, items, and metadata.
        use_metadata: bool, optional
            Whether to use item metadata. Default is True.
        use_batch_norm: bool, optional
            Whether to include BatchNorm1d layers after each hidden linear layer
            (before ReLU activation). Default is True.
        hidden_layers: List[int], optional
            A list of integers defining the sizes of the hidden layers. 
            For example, `[512, 256, 128]` creates three hidden layers.
            If None, defaults to `[1024, 128]`. Default is None.
        use_cuda: bool, optional
            Whether to use CUDA if available. Default is False.
        """
        super(MLP, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_batch_norm = use_batch_norm
        self.hidden_layers = hidden_layers if hidden_layers is not None else [1024, 128] # Store or default
        self.use_cuda = use_cuda
        
        self.input_shape = self.n_factors * 2

        if use_metadata:
            self.n_distinct_metadata = len(self.n_metadata.keys())
            self.input_shape += (self.n_factors * self.n_distinct_metadata)

        self.user = ScaledEmbedding(self.n_users, self.n_factors, sparse=True) 
        self.item = ScaledEmbedding(self.n_items, self.n_factors, sparse=True) 
        
        if use_metadata:
            self.metadata_embeddings = torch.nn.ModuleList( # Renamed from self.metadata to avoid confusion
                                [ScaledEmbedding(size, self.n_factors, sparse=True) for _ , size in self.n_metadata.items()])

        # Dynamically create layers
        self.fcs = torch.nn.ModuleList()
        if self.use_batch_norm:
            self.bns = torch.nn.ModuleList()

        current_input_size = self.input_shape
        for layer_size in self.hidden_layers:
            self.fcs.append(torch.nn.Linear(current_input_size, layer_size))
            if self.use_batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(layer_size))
            current_input_size = layer_size
        
        self.output_layer = torch.nn.Linear(current_input_size, 1) # Final prediction layer


    def forward(self, batch, user_key, item_key, metadata_key=None):

        user = batch[user_key]
        item = batch[item_key]

        user_embedding = self.user(user)
        item_embedding = self.item(item)
        
        #### metadata
        if self.use_metadata:
            metadata_vals = batch[metadata_key] # Corrected variable name
            
            for idx, layer in enumerate(self.metadata_embeddings): # Use renamed metadata_embeddings
                single_layer = layer(metadata_vals[:, idx]) # Use metadata_vals
                item_embedding = torch.cat([item_embedding, single_layer], axis=1)
        ###

        net = torch.cat([user_embedding, item_embedding], axis=1)

        for i in range(len(self.fcs)):
            net = self.fcs[i](net)
            if self.use_batch_norm:
                net = self.bns[i](net)
            net = torch.nn.functional.relu(net)
        
        net = self.output_layer(net)
        
        return net