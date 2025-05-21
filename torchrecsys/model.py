# -*- coding: utf-8 -*-

import torch
import json
import torch.profiler # Added for profiling
from torchrecsys.dataset.dataset import ProcessData, FastDataLoader
from torchrecsys.collaborative.linear import Linear
from torchrecsys.collaborative.mlp import MLP
from torchrecsys.collaborative.fm import FM
from torchrecsys.helper.cuda import gpu
from torchrecsys.helper.loss import hinge_loss
from torchrecsys.helper.evaluate import auc_score
from torchrecsys.evaluate.metrics import Metrics
import pandas as pd
from typing import List


class TorchRecSys(torch.nn.Module):
    
    """
    Main class for TorchRecSys models, orchestrating data processing, model initialization,
    training, evaluation, and prediction.

    Encodes users (or item sequences) and items in a low dimensional space, 
    using dot products as similarity measure for Collaborative Filtering models.
    
    Parameters:
    -----------
    dataset: pd.DataFrame
        Input pandas DataFrame containing user-item interactions and optional metadata.
    user_id_col: str
        Column name in `dataset` that represents user IDs.
    item_id_col: str
        Column name in `dataset` that represents item IDs.
    n_factors: int, optional
        Dimensionality of the embedding space for users and items. Default is 80.
    net_type: str, optional
        Type of the underlying recommendation model. Supported values: 'linear', 'mlp', 'fm'.
        Default is 'linear'.
    metadata_id_col: List[str], optional
        List of column names in `dataset` that represent item metadata features. 
        Each metadata column is treated as a separate categorical feature. Default is None.
    split_ratio: float, optional
        Ratio for splitting the dataset into training and testing sets. 
        Value should be between 0 and 1. Default is 0.8 (80% train, 20% test).
    dynamic_neg_sampling: bool, optional
        If True, negative item samples are generated on-the-fly during training.
        If False, fixed negative samples are generated once during data pre-processing.
        Default is False.
    use_amp: bool, optional
        If True and `use_cuda` is True, enables Automatic Mixed Precision (AMP) for training,
        which can speed up training and reduce memory usage on compatible GPUs. Default is False.
    use_cuda: bool, optional
        If True, the model and data will be moved to a CUDA-enabled GPU for training and inference.
        Default is False.
    debug: bool, optional
        If True, enables debug mode (currently not extensively used but reserved for future).
        Default is False.
    path: str, optional
        Path for saving auxiliary files like `config.json` or `meta.csv` (if metadata is used).
        Primarily for legacy purposes as data is now processed in memory. Default is './'.
    """
    
    def __init__(self,
                 dataset: pd.DataFrame,
                 user_id_col: str,
                 item_id_col: str,
                 n_factors: int = 80, 
                 net_type: str = 'linear', 
                 metadata_id_col: List[str] = None,
                 split_ratio: float = 0.8,
                 dynamic_neg_sampling: bool = False, 
                 use_amp: bool = False, 
                 use_cuda: bool = False,
                 debug: bool = False,
                 path: str = './'):

        super().__init__()
        
        self.path = path # For saving config and meta if still desired
        self.dynamic_neg_sampling = dynamic_neg_sampling # Store
        self.use_amp = use_amp # Store AMP flag
        self.use_cuda = use_cuda # Store CUDA flag early for GradScaler
        # self.profile_epochs = 0 # Optional: init here, or just use param in fit

        self.grad_scaler = None
        if self.use_amp and self.use_cuda:
            self.grad_scaler = torch.cuda.amp.GradScaler()

        self.data_processor = ProcessData(dataset=dataset,
                                          user_id_col=user_id_col,
                                          item_id_col=item_id_col,
                                          metadata_id_col=metadata_id_col,
                                          split_ratio=split_ratio,
                                          dynamic_neg_sampling=self.dynamic_neg_sampling) # Pass down
        
        self.data_processor.prepare_data()
        # self.data_processor.write_data(self.path) # Optionally write config/meta

        self.config = self.data_processor.config
        self.n_users = self.config.get('num_users')
        self.n_items = self.config.get('num_items')      
        self.metadata_size = self.config.get('num_metadata')
        
        # Store metadata column names if they exist for FastDataLoader
        self.metadata_name = metadata_id_col if hasattr(self.data_processor, 'metadata_id_col') else None


        self.n_factors = n_factors
        # self.use_cuda = use_cuda # Already set above
        self.net_type = net_type
        
        self.use_metadata = True if self.metadata_name else False

        self.debug = debug

        self._init_net(net_type=net_type)

    # _read_metadata is no longer needed as config comes from ProcessData
    # def _read_metadata(self, path):
    #     with open(f'{path}/config.json') as json_file:    
    #         config = json.load(json_file)
    #     return config

    def _init_net(self, net_type='linear'):

        assert net_type in ('linear', 'mlp', 'neucf', 'fm', 'lstm'), 'Net type must be one of "linear", "mlp", "neu", "ease" or "lstm"'

        if net_type == 'linear':

          print('Linear Collaborative Filtering')

          self.net = Linear(n_users=self.n_users, 
                            n_items=self.n_items, 
                            n_metadata=self.metadata_size, 
                            n_factors=self.n_factors, 
                            use_metadata=self.use_metadata, 
                            use_cuda=self.use_cuda)
        
        elif net_type == 'mlp':

            print('Multi Layer Perceptron')

            self.net = MLP(n_users=self.n_users, 
                           n_items=self.n_items, 
                           n_metadata=self.metadata_size, 
                           n_factors=self.n_factors, 
                           use_metadata=self.use_metadata, 
                           use_cuda=self.use_cuda)
          
        elif net_type == 'fm':

            print('Factorization Machine')

            self.net = FM(n_users=self.n_users, 
                          n_items=self.n_items, 
                          n_metadata=self.metadata_size, 
                          n_factors=self.n_factors, 
                          use_metadata=self.use_metadata, 
                          use_cuda=self.use_cuda)
          
        elif net_type == 'neucf':
            NotImplementedError('NeuCF not implemented yet')

        elif net_type == 'lstm':
            NotImplementedError('LSTM not implemented yet')

        self.net = gpu(self.net, self.use_cuda)


    def forward(self, net, batch):

        positive_score = gpu(net.forward(batch, 
                                         user_key='user_id', 
                                         item_key='pos_item_id',
                                         metadata_key='pos_metadata_id'),
                             self.use_cuda)

        negative_score = gpu(net.forward(batch, 
                                         user_key='user_id', 
                                         item_key='neg_item_id',
                                         metadata_key='neg_metadata_id'),
                             self.use_cuda)

        return positive_score, negative_score
    
    
    def backward(self, loss_value, optimizer): # Changed to accept loss_value directly
                
        optimizer.zero_grad()
        
        if self.use_amp and self.use_cuda and self.grad_scaler is not None:
            self.grad_scaler.scale(loss_value).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss_value.backward()
            optimizer.step()

        return loss_value.item()
    
      
    def fit(self, optimizer, epochs=10, batch_size=512, profile_epochs: int = 0):
        """
        Fits the model to the training data using the specified optimizer.

        Parameters:
        -----------
        optimizer: torch.optim.Optimizer
            The optimizer (e.g., `torch.optim.Adam`) to use for training.
        epochs: int, optional
            Number of epochs to train the model. Default is 10.
        batch_size: int, optional
            Number of samples per batch during training. Default is 512.
        profile_epochs: int, optional
            If set to a value greater than 0 (e.g., 1), the first epoch of training
            will be profiled using `torch.profiler`. A summary table of performance
            metrics (CPU time, CUDA time, memory usage) will be printed to the console.
            This is useful for identifying performance bottlenecks. For more detailed
            analysis, such as generating TensorBoard traces, users can adapt the
            profiler setup within this method. Default is 0 (no profiling).
        """
        train_loader = FastDataLoader(data=self.data_processor.train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      dynamic_neg_sampling=self.dynamic_neg_sampling,
                                      n_items=self.n_items,
                                      item_to_metadata_map=self.data_processor.item_to_metadata_map,
                                      metadata_id_cols=self.metadata_name) # self.metadata_name stores original list of meta id cols
        
        for epoch in range(epochs):
            self.net = self.net.train()
            total_loss = 0
            num_batches = 0

            # Simpler Console Profiling for the first epoch if profile_epochs > 0
            if profile_epochs > 0 and epoch == 0:
                print(f"\n--- Starting Profiling for Epoch {epoch+1} ---")
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.use_cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                    # For more detailed step-by-step or TensorBoard, use schedule and on_trace_ready:
                    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    # on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/epoch_{epoch}')
                ) as prof:
                    for batch_idx, batch in enumerate(train_loader):
                        num_batches += 1
                        mini_batch = {}
                        for key, values in batch.items():
                            mini_batch.update({key: gpu(values, self.use_cuda)})
                        
                        with torch.cuda.amp.autocast(enabled=(self.use_amp and self.use_cuda)):
                            positive, negative = self.forward(net=self.net, batch=mini_batch)
                            loss_value_tensor = hinge_loss(positive, negative)
                        
                        current_loss_item = self.backward(loss_value_tensor, optimizer)
                        total_loss += current_loss_item
                        # if profiler schedule is used, call prof.step() here
                        # if batch_idx >= (1 + 1 + 3) * 1: # Example: wait + warmup + active * repeat
                        #    break # Stop after a few profiled steps if using schedule for quick check

                print("--- Profiler Results (First Epoch) ---")
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
                # If only profiling one epoch for console, and it's done, maybe break or set profile_epochs to 0
                # For this subtask, we'll let other epochs run normally.

            else: # Normal epoch execution (no profiling or subsequent epochs)
                for batch in train_loader:
                    num_batches += 1
                    mini_batch = {}
                    for key, values in batch.items():
                        mini_batch.update({key: gpu(values, self.use_cuda)})
                    
                    with torch.cuda.amp.autocast(enabled=(self.use_amp and self.use_cuda)):
                        positive, negative = self.forward(net=self.net, batch=mini_batch)
                        loss_value_tensor = hinge_loss(positive, negative)
                    
                    current_loss_item = self.backward(loss_value_tensor, optimizer)
                    total_loss += current_loss_item

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f'|--- Epoch {epoch+1}/{epochs} --- Training Loss: {avg_loss:.4f}')
            # Optional: AUC calculation here if needed, but kept out for simplicity as per previous refactors


    def evaluate(self, batch_size=512, eval_metrics=['loss', 'auc']):

        self.net = self.net.eval()
        measures = Metrics()
        
        all_metrics_results = {metric: [] for metric in eval_metrics}

        if not self.data_processor.test_data.get('user_id', torch.empty(0)).numel() == 0 : # Check if test_data is not empty
            test_loader = FastDataLoader(data=self.data_processor.test_data,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         dynamic_neg_sampling=self.dynamic_neg_sampling, # Using same flag for eval for now
                                         n_items=self.n_items,
                                         item_to_metadata_map=self.data_processor.item_to_metadata_map,
                                         metadata_id_cols=self.metadata_name)

            for batch in test_loader:
                mini_batch = {}
                for key, values in batch.items():
                    mini_batch.update({key: gpu(values, self.use_cuda)}) # Data to CUDA

                # AMP: Wrap forward pass and loss calculation with autocast
                with torch.cuda.amp.autocast(enabled=(self.use_amp and self.use_cuda)):
                    positive_test, negative_test = self.forward(net=self.net, batch=mini_batch)
                    if 'loss' in eval_metrics: 
                        loss_val_tensor = hinge_loss(positive_test, negative_test)
                        all_metrics_results['loss'].append(loss_val_tensor.item() if hasattr(loss_val_tensor, 'item') else loss_val_tensor)

                # AUC calculation does not need to be in autocast context if it's just based on scores
                if 'auc' in eval_metrics:
                    # Ensure positive_test and negative_test are float for auc_score if they are not already
                    auc_val = measures.auc_score(positive_test.float(), negative_test.float())
                    all_metrics_results['auc'].append(auc_val.item() if hasattr(auc_val, 'item') else auc_val)
                
                # Add other metrics like hit_rate if needed and calculated

            avg_metrics = {}
            for metric_name, values in all_metrics_results.items():
                if values:
                    avg_metrics[metric_name] = sum(values) / len(values)
                else:
                    avg_metrics[metric_name] = 0 # Or appropriate default

            for metric, value in avg_metrics.items():
                print(f'|--- Testing {metric}: {value:.4f}')
        else:
            print("|--- No test data to evaluate.")


    def predict(self, user_id: int, top_k: int = 10, prediction_batch_size: int = 4096):
        """
        Generates top-K item recommendations for a given user.

        Scores all items for the user in batches to manage memory efficiently,
        then returns the indices of the items with the highest scores.

        Parameters:
        -----------
        user_id: int
            The ID of the user for whom to generate recommendations.
        top_k: int, optional
            The number of top items to recommend. Default is 10.
        prediction_batch_size: int, optional
            The batch size used internally for scoring items during prediction.
            Adjust this based on available memory (especially GPU memory if `use_cuda=True`).
            Default is 4096.

        Returns:
        --------
        torch.Tensor
            A 1D tensor containing the indices (IDs) of the top-K recommended items,
            sorted by predicted score in descending order.
        """
        self.net = self.net.eval()
        
        all_item_ids = list(range(self.n_items))
        all_scores_list = []

        # Pre-fetch and prepare metadata_df if it's going to be used
        meta_df_for_lookup = None
        if self.use_metadata and self.data_processor.meta_data_df is not None:
            meta_df_for_lookup = self.data_processor.meta_data_df.copy()
            # Ensure all expected metadata columns are present
            if not all(col in meta_df_for_lookup.columns for col in self.metadata_name):
                raise ValueError(
                    "Not all metadata columns specified in self.metadata_name are present in self.data_processor.meta_data_df."
                )
            # Set pos_item_id as index for potentially faster lookup, though merge is still used.
            # meta_df_for_lookup = meta_df_for_lookup.set_index('pos_item_id')


        for i in range(0, self.n_items, prediction_batch_size):
            item_id_chunk = all_item_ids[i:i + prediction_batch_size]
            
            current_batch_df = pd.DataFrame({
                'user_id': [user_id] * len(item_id_chunk),
                'pos_item_id': item_id_chunk
            })

            if self.use_metadata and meta_df_for_lookup is not None:
                # Merge current chunk with relevant metadata
                # Only select necessary columns from meta_df_for_lookup for the merge
                current_batch_df = pd.merge(current_batch_df, 
                                            meta_df_for_lookup[['pos_item_id'] + self.metadata_name], 
                                            on='pos_item_id', 
                                            how='left')
                
                for col in self.metadata_name:
                    # Ensure missing metadata is an empty list after merge for consistent processing
                    current_batch_df[col] = current_batch_df[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else x))


            # Convert chunk to tensors
            input_batch_chunk = {
                'user_id': torch.tensor(current_batch_df['user_id'].values).long(),
                'pos_item_id': torch.tensor(current_batch_df['pos_item_id'].values).long()
            }

            if self.use_metadata:
                pos_metadata_tensors_chunk = []
                for meta_col in self.metadata_name:
                    # Ensure column exists, can happen if a chunk had no items with this metadata type and it was all NaN
                    if meta_col in current_batch_df:
                         meta_series = current_batch_df[meta_col].apply(lambda x: x if isinstance(x, list) else [])
                         padded_meta = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in meta_series], batch_first=True, padding_value=0).long()
                         pos_metadata_tensors_chunk.append(padded_meta)
                    else: # Should not happen if fillna([]) was effective
                         # Create empty tensor of appropriate shape if a metadata column is entirely missing for a chunk
                         # This is a fallback, ideally fillna handles this.
                         empty_meta_for_chunk = torch.zeros(len(item_id_chunk), 0, dtype=torch.long) # Assuming 0 is padding_value
                         pos_metadata_tensors_chunk.append(empty_meta_for_chunk)


                if len(pos_metadata_tensors_chunk) > 0:
                    input_batch_chunk['pos_metadata_id'] = torch.stack(pos_metadata_tensors_chunk, dim=1) if len(pos_metadata_tensors_chunk) > 1 else pos_metadata_tensors_chunk[0]
                else: # No metadata features, or all were empty
                    input_batch_chunk['pos_metadata_id'] = torch.empty(len(item_id_chunk), 0, dtype=torch.long) 
            
            # Move chunk to CUDA
            for key, values in input_batch_chunk.items():
                input_batch_chunk[key] = gpu(values, self.use_cuda)

            # Get scores for the chunk
            with torch.cuda.amp.autocast(enabled=(self.use_amp and self.use_cuda)):
                score_chunk = self.net.forward(input_batch_chunk, 
                                               user_key='user_id', 
                                               item_key='pos_item_id',
                                               metadata_key='pos_metadata_id' if self.use_metadata else None)
            all_scores_list.append(score_chunk.cpu()) # Move to CPU to free GPU mem

        # Concatenate all scores
        all_scores = torch.cat(all_scores_list, dim=0)
        
        # Sort scores to get top_k items
        # Ensure all_scores is float for sorting if it's not already (e.g. if model output was half)
        sorted_scores, sorted_indices = torch.sort(all_scores.squeeze().float(), descending=True)
        
        # The sorted_indices correspond to the original item IDs because we processed items in order (0 to n_items-1)
        top_k_indices = sorted_indices[:top_k]
        
        return top_k_indices

    # _create_inference_batch is refactored into predict()
    # _get_metadata is no longer needed as meta_data is in self.data_processor