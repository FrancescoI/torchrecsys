import torch
from torchrecsys.embeddings.init_embeddings import ScaledEmbedding
from torchrecsys.helper.cuda import gpu

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
    
    def __init__(self, 
                 dataloader,
                 n_users, 
                 n_items, 
                 n_metadata, 
                 n_factors, 
                 use_metadata=True, 
                 use_cuda=False):
        
        super(FM, self).__init__()

        self.dataloader = dataloader

        self.n_users = n_users
        self.n_items = n_items
        self.n_metadata = n_metadata
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda

        self.n_input = self.n_users + self.n_items
        
        self.user = ScaledEmbedding(self.n_users, self.n_factors)
        self.item = ScaledEmbedding(self.n_items, self.n_factors)
                
        self.linear_user = ScaledEmbedding(self.n_users, 1)
        self.linear_item = ScaledEmbedding(self.n_items, 1)

        if use_metadata:

            self.n_distinct_metadata = len(self.n_metadata.keys())
            
            self.metadata = torch.nn.ModuleList(
                                [ScaledEmbedding(size, self.n_factors) for _ , size in self.n_metadata.items()])

            self.linear_metadata = torch.nn.ModuleList(
                                [ScaledEmbedding(size, 1) for _ , size in self.n_metadata.items()])



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


    def predict(self, user_ids, top_k=None):

        """
        It returns sorted item indexes for a given user.
        """

        num_users = len(user_ids)
        
        # LINEAR
        user_ids = gpu(torch.tensor(user_ids), self.use_cuda)
        user_linear = self.linear_user(user_ids).reshape(num_users, 1, 1)
        user_linear = user_linear.repeat(1, self.n_items, 1)

        item_linear = self.linear_item.weight.data
        
        if self.use_metadata:
            mapping = gpu(torch.from_numpy(self.dataloader.map_item_metadata()), self.use_cuda)
            
            for idx, layer in enumerate(self.linear_metadata):
                single_layer = layer.weight.data
                item_meta_linear = torch.cat([item_meta_linear, single_layer], axis=0) if idx > 0 else single_layer

            item_meta_linear = torch.cat([item_linear, item_meta_linear], axis=0)

            mapping = mapping.reshape(mapping.shape[0], 1, mapping.shape[1]) # (n_items X (n_items + n_metadata))
            item_meta_linear = item_meta_linear.transpose(1, 0).reshape(1, 1, mapping.shape[2]) # (1 X n_factors X (n_items + n_metadata))

            cat = mapping * item_meta_linear
            item_linear = cat[cat != 0].reshape(self.n_items, 1, 1+self.n_distinct_metadata)
            item_linear = item_linear.transpose(2,1)
            item_linear = item_linear.reshape(self.n_items, 1 * (1+self.n_distinct_metadata))

        item_linear = item_linear.reshape(1, self.n_items, 1 * (1+self.n_distinct_metadata)).repeat(num_users, 1, 1)
        linear = torch.cat([user_linear, item_linear], dim=2)
        linear = linear.sum(2)

        # EMBEDDING
        user_embedding = self.user(torch.tensor(user_ids)).reshape(num_users, 1, self.n_factors)
        user_embedding = user_embedding.repeat(1, self.n_items, 1) 
        
        item_embedding = self.item.weight.data
        
        if self.use_metadata:
            mapping = gpu(torch.from_numpy(self.dataloader.map_item_metadata()), self.use_cuda)
            
            for idx, layer in enumerate(self.metadata):
                single_layer = layer.weight.data
                item_meta_embedding = torch.cat([item_meta_embedding, single_layer], axis=0) if idx > 0 else single_layer

            item_meta_embedding = torch.cat([item_embedding, item_meta_embedding], axis=0)

            mapping = mapping.reshape(mapping.shape[0], 1, mapping.shape[1]) # (n_items X (n_items + n_metadata))
            item_meta_embedding = item_meta_embedding.transpose(1, 0).reshape(1, self.n_factors, mapping.shape[2]) # (1 X n_factors X (n_items + n_metadata))

            cat = mapping * item_meta_embedding
            item_embedding = cat[cat != 0].reshape(self.n_items, self.n_factors, 1+self.n_distinct_metadata)
            item_embedding = item_embedding.transpose(2,1)
            item_embedding = item_embedding.reshape(self.n_items, self.n_factors * (1+self.n_distinct_metadata))

        item_embedding = item_embedding.reshape(1, self.n_items, self.n_factors * (1+self.n_distinct_metadata)) 
        item_embedding = item_embedding.repeat(num_users, 1, 1)

        embedding = torch.cat([user_embedding, item_embedding], dim=2)

        power_of_sum = embedding.sum(dim=2).pow(2)
        sum_of_power = embedding.pow(2).sum(dim=2)

        pairwise = (power_of_sum - sum_of_power) * 0.5

        ### COMBINE
        prediction = torch.sigmoid(linear + pairwise)

        sorted_index = torch.argsort(prediction, dim=0, descending=True)

        if top_k:
            sorted_index = sorted_index[:, :top_k].squeeze()

        return sorted_index