# -*- coding: utf-8 -*-

from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import ast
import json




class Data:

    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str,
                 metadata_id_col: List[str] = None,
                 split_ratio: float = 0.8,
                 dynamic_neg_sampling: bool = False): # Added for ProcessData to pass down
        
        self.dataset = dataset

        self.user_id = user_id_col
        self.item_id = item_id_col

        self.num_items = len(self.dataset[self.item_id].unique())
        self.num_users = len(self.dataset[self.user_id].unique())
        
        self.dynamic_neg_sampling = dynamic_neg_sampling # Store for conditional logic

        if not self.dynamic_neg_sampling:
            self.dataset = self._get_negative_items(self.dataset)

        self.split_ratio = split_ratio
        
        if metadata_id_col:
            self.metadata_id = metadata_id_col
            if not self.dynamic_neg_sampling: # Only add fixed negative metadata if not dynamic
                self.dataset = self._add_negative_metadata(self.dataset)
            self.negative_metadata_id = self._get_negative_metadata_column_names() # Still need this for column naming conventions
            self.metadata_size = self._get_metadata_size(self.metadata_id) # Metadata size is independent of sampling

    def _get_metadata_size(self, metadata_id_col):
        
        metadata_size = {}

        for col in metadata_id_col:
            metadata_size.update({col: len(self.dataset[col].unique())})

        return metadata_size
  
    def _get_negative_items(self, dataset):

        negative_items = np.random.randint(low=0, 
                                           high=self.num_items, 
                                           size=self.dataset.shape[0])

        dataset['neg_item'] = negative_items

        return dataset

    def _get_metadata(self, dataset):

        metadata = (dataset
                    .set_index(self.item_id)[self.metadata_id]
                    .reset_index()
                    .drop_duplicates())

        return metadata
                        
    def _add_negative_metadata(self, dataset):
        
        metadata = self._get_metadata(dataset)

        metadata_negative_names = self._get_negative_metadata_column_names()

        metadata.columns = ['neg_item'] + metadata_negative_names

        dataset = pd.merge(dataset, metadata, on='neg_item')

        return dataset

    def _get_negative_metadata_column_names(self):

        return ['neg_' + metadata_column for metadata_column in self.metadata_id]

    def _apply_negative_metadata(self):

        metadata = self._get_negative_metadata()

        dataset = pd.merge(self.dataset, metadata, on='neg_item')

        return dataset

    def map_item_metadata(self):

        grouping_col = [self.item_id] + self.metadata_id

        # Sort by the item identifier to keep items in a consistent order
        data = (
            self.dataset
            .groupby(grouping_col, as_index=False)
            .agg({self.user_id: 'count'})
            .sort_values(self.item_id)
        )

        dummies = None
        for metadata in grouping_col:
            dummy = pd.get_dummies(data[metadata])
            dummies = pd.concat([dummies, dummy], axis=1) if dummies is not None else dummy

        mapping = dummies.values

        return mapping

                        
class ProcessData(Data):

    def __init__(self, 
                 dataset: pd.DataFrame, 
                 user_id_col: str, 
                 item_id_col: str,
                 metadata_id_col: List[str] = None,
                 split_ratio: float = 0.9,
                 dynamic_neg_sampling: bool = False): # Added

        super().__init__(dataset, user_id_col, item_id_col, metadata_id_col, split_ratio, dynamic_neg_sampling) # Pass down

        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.dynamic_neg_sampling = dynamic_neg_sampling # Store
        
        if metadata_id_col:
            self.metadata_id_col = metadata_id_col

    def prepare_data(self):
        """
        Processes the data and stores it in memory.
        """
        # Define columns to select based on dynamic_neg_sampling
        cols = [self.user_id_col, self.item_id_col]
        new_cols = ['user_id', 'pos_item_id']

        if not self.dynamic_neg_sampling:
            cols.append('neg_item')
            new_cols.append('neg_item_id')

        if hasattr(self, 'metadata_id_col') and self.metadata_id_col:
            cols.extend(self.metadata_id_col) # Positive metadata
            new_cols.extend(self.metadata_id_col)
            if not self.dynamic_neg_sampling: # Negative metadata only if not dynamic
                cols.extend(['neg_' + meta for meta in self.metadata_id_col])
                new_cols.extend(['neg_' + meta for meta in self.metadata_id_col])
            
        # Select and rename columns
        # Ensure all selected columns exist in self.dataset before copying
        existing_cols_in_dataset = [col for col in cols if col in self.dataset.columns]
        dataset = self.dataset[existing_cols_in_dataset].copy()
        
        # Map new_cols to the existing columns in the copied dataset
        # The number of new_cols should match the number of existing_cols_in_dataset
        current_new_cols_map = {}
        temp_new_cols = ['user_id', 'pos_item_id'] # Base columns
        if not self.dynamic_neg_sampling:
            temp_new_cols.append('neg_item_id')
        
        pos_meta_cols_ordered = []
        if hasattr(self, 'metadata_id_col') and self.metadata_id_col:
            for meta_col in self.metadata_id_col:
                if meta_col in existing_cols_in_dataset: # Check if positive metadata column was actually selected
                    pos_meta_cols_ordered.append(meta_col)
            temp_new_cols.extend(pos_meta_cols_ordered)

            if not self.dynamic_neg_sampling:
                neg_meta_cols_ordered = []
                for meta_col in self.metadata_id_col:
                     # Check if negative metadata column was actually selected
                    if 'neg_' + meta_col in existing_cols_in_dataset:
                        neg_meta_cols_ordered.append('neg_' + meta_col)
                temp_new_cols.extend(neg_meta_cols_ordered)
        
        dataset.columns = temp_new_cols


        # Process metadata values (e.g., string to list)
        if hasattr(self, 'metadata_id_col') and self.metadata_id_col:
            for col in self.metadata_id_col: # For positive metadata
                 if col in dataset.columns: # Ensure column exists after selection
                    dataset[col] = dataset[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
            if not self.dynamic_neg_sampling: # For negative metadata
                for col in ['neg_' + meta for meta in self.metadata_id_col]:
                    if col in dataset.columns: # Ensure column exists
                        dataset[col] = dataset[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
        
        self.config = {
            'num_users': self.num_users,
            'num_items': self.num_items, # n_items from Data class
            'num_metadata': self.metadata_size if hasattr(self, 'metadata_size') else {}
        }

        self.item_to_metadata_map = None
        if hasattr(self, 'metadata_id_col') and self.metadata_id_col:
            # Use existing _get_metadata, but ensure it uses the original item_id column name
            # And it should use self.dataset BEFORE it's potentially filtered by 'cols' above if 'neg_item' was used.
            # For simplicity, let's assume self.dataset still has original item IDs and metadata.
            # _get_metadata expects self.item_id and self.metadata_id (which are original column names)
            meta_df_for_lookup = super()._get_metadata(self.dataset) # Gets unique item-metadata pairs
            meta_df_for_lookup.columns = ['pos_item_id'] + self.metadata_id_col # Rename for consistency
            
            self.meta_data_df = meta_df_for_lookup.copy() # For writing to CSV if needed, and for TorchRecSys.predict

            # Create item_to_metadata_map for FastDataLoader (if dynamic sampling)
            self.item_to_metadata_map = {}
            for _, row in self.meta_data_df.iterrows():
                item_id_val = row['pos_item_id']
                meta_dict = {}
                for meta_col_original in self.metadata_id_col:
                    # Ensure metadata is list, apply literal_eval if string
                    meta_val = row[meta_col_original]
                    if isinstance(meta_val, str):
                        try:
                            meta_val = ast.literal_eval(meta_val)
                        except (ValueError, SyntaxError):
                            meta_val = [] # Default to empty list if eval fails
                    meta_dict[meta_col_original] = meta_val if isinstance(meta_val, list) else []
                self.item_to_metadata_map[item_id_val] = meta_dict
        else:
            self.meta_data_df = None
            self.item_to_metadata_map = None


        # Split data
        df_train, df_test = None, None
        # Split data using the processed 'dataset' DataFrame
        if self.split_ratio < 1:
            df_train, df_test = train_test_split(dataset, test_size=1-self.split_ratio, random_state=42)
        else:
            df_train = dataset
            # Ensure df_test has the same columns as df_train if it's empty
            df_test = pd.DataFrame(columns=dataset.columns if not dataset.empty else ['user_id', 'pos_item_id'])


        # Convert to dictionary of tensors
        self.train_data = self._convert_df_to_tensor_dict(df_train)
        self.test_data = self._convert_df_to_tensor_dict(df_test)

    def _convert_df_to_tensor_dict(self, df: pd.DataFrame):
        if df.empty:
            # Return structure with empty tensors if df is empty, matching expected keys
            empty_dict = {
                'user_id': torch.empty(0, dtype=torch.long),
                'pos_item_id': torch.empty(0, dtype=torch.long),
            }
            if not self.dynamic_neg_sampling:
                empty_dict['neg_item_id'] = torch.empty(0, dtype=torch.long)
            
            if hasattr(self, 'metadata_id_col') and self.metadata_id_col:
                empty_dict['pos_metadata_id'] = torch.empty(0, dtype=torch.long) # Placeholder, structure might vary
                if not self.dynamic_neg_sampling:
                     empty_dict['neg_metadata_id'] = torch.empty(0, dtype=torch.long) # Placeholder
            return empty_dict

        tensor_dict = {
            'user_id': torch.from_numpy(df['user_id'].values).long(),
            'pos_item_id': torch.from_numpy(df['pos_item_id'].values).long(),
        }
        if not self.dynamic_neg_sampling:
            tensor_dict['neg_item_id'] = torch.from_numpy(df['neg_item_id'].values).long()


        if hasattr(self, 'metadata_id_col') and self.metadata_id_col:
            pos_meta_tensors_list = []
            # Check if all metadata columns are actually in df.columns before processing
            # This is important because earlier filtering might remove them if they were all NaN or something.
            valid_pos_meta_cols = [mc for mc in self.metadata_id_col if mc in df.columns]

            for meta_col in valid_pos_meta_cols:
                pos_sequences = [torch.tensor(seq, dtype=torch.long) for seq in df[meta_col]]
                pos_padded = torch.nn.utils.rnn.pad_sequence(pos_sequences, batch_first=True, padding_value=0)
                pos_meta_tensors_list.append(pos_padded)
            
            if len(pos_meta_tensors_list) > 0:
                tensor_dict['pos_metadata_id'] = torch.stack(pos_meta_tensors_list, dim=1) if len(pos_meta_tensors_list) > 1 else pos_meta_tensors_list[0]
            # else:
                # tensor_dict['pos_metadata_id'] = torch.empty(df.shape[0], 0, dtype=torch.long) # Or some other indicator of missing metadata


            if not self.dynamic_neg_sampling:
                neg_meta_tensors_list = []
                valid_neg_meta_cols = ['neg_' + mc for mc in self.metadata_id_col if 'neg_' + mc in df.columns]
                for meta_col_neg in valid_neg_meta_cols: # e.g. 'neg_genre_ids'
                    neg_sequences = [torch.tensor(seq, dtype=torch.long) for seq in df[meta_col_neg]]
                    neg_padded = torch.nn.utils.rnn.pad_sequence(neg_sequences, batch_first=True, padding_value=0)
                    neg_meta_tensors_list.append(neg_padded)

                if len(neg_meta_tensors_list) > 0:
                    tensor_dict['neg_metadata_id'] = torch.stack(neg_meta_tensors_list, dim=1) if len(neg_meta_tensors_list) > 1 else neg_meta_tensors_list[0]
                # else:
                    # tensor_dict['neg_metadata_id'] = torch.empty(df.shape[0], 0, dtype=torch.long)

        return tensor_dict

    def write_data(self, path: str):
        """
        Writes config.json and meta.csv (if applicable).
        Train/test data (now tensors) are kept in memory.
        """
        with open(f'{path}/config.json', 'w') as file:
            json.dump(self.config, file)

        if self.meta_data_df is not None: # Use self.meta_data_df
            self.meta_data_df.to_csv(f'{path}/meta.csv', index=False)


class FastDataLoader:
    
    def __init__(self, 
                 data: dict, # Expects a dictionary of tensors
                 batch_size: int = 32,
                 shuffle: bool = False,
                 dynamic_neg_sampling: bool = False, # Added
                 n_items: int = None, # Added - total number of items for sampling
                 item_to_metadata_map: dict = None, # Added - map from item_id to its metadata features
                 metadata_id_cols: List[str] = None): # Added - list of original metadata column names (e.g. ['genre_ids'])
        """
        Initialize a FastDataLoader.
        :param data: pre-processed data (dictionary of tensors).
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data.
        :param dynamic_neg_sampling: if True, sample negatives on the fly.
        :param n_items: total number of unique items. Required if dynamic_neg_sampling is True.
        :param item_to_metadata_map: dict mapping item_id to its metadata. Required for dynamic negatives with metadata.
        :param metadata_id_cols: list of original metadata column names. Required for dynamic negatives with metadata.
        """
        self.data = data 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dynamic_neg_sampling = dynamic_neg_sampling
        self.n_items = n_items
        self.item_to_metadata_map = item_to_metadata_map
        self.metadata_id_cols = metadata_id_cols # e.g. ['genre_ids', 'category_ids']

        if self.dynamic_neg_sampling and self.n_items is None:
            raise ValueError("n_items must be provided for dynamic negative sampling.")
        if self.dynamic_neg_sampling and self.metadata_id_cols and self.item_to_metadata_map is None:
            # Only raise error if metadata_id_cols is also provided, implying metadata is expected
            raise ValueError("item_to_metadata_map must be provided for dynamic negative sampling with metadata.")


        self.dataset_len = 0
        # Check if data is not empty and 'user_id' tensor exists
        if 'user_id' in self.data and isinstance(self.data['user_id'], torch.Tensor) and self.data['user_id'].shape[0] > 0:
             self.dataset_len = self.data['user_id'].shape[0]
        
        if self.shuffle and self.dataset_len > 0:
            self.shuffle_indices() # Initial shuffle

        self.num_batches = int(np.ceil(self.dataset_len / self.batch_size)) if self.dataset_len > 0 else 0
        
    def shuffle_indices(self):
        """Shuffles the indices for data iteration."""
        if self.dataset_len == 0: return # Cannot shuffle empty dataset
        self.indices = torch.randperm(self.dataset_len)

    def __iter__(self):
        self.i = 0
        if self.shuffle and self.dataset_len > 0 : # Re-shuffle at the start of each epoch
            self.shuffle_indices()
        return self

    def _get_padded_metadata_tensor(self, item_ids_list: List[int]):
        """
        For a list of item_ids, fetches their metadata, pads, and stacks them.
        Returns a tensor similar to how 'pos_metadata_id' or 'neg_metadata_id' are structured.
        """
        if not self.item_to_metadata_map or not self.metadata_id_cols:
            return None # Or torch.empty(len(item_ids_list), 0, dtype=torch.long) if a specific shape is needed

        # Each element in all_items_meta_sequences will be a list of tensors, one per metadata feature
        all_items_meta_tensors_stacked = []

        # This outer loop iterates through each metadata type (e.g., genre, category)
        meta_feature_tensors_for_current_batch = []
        for meta_col_original in self.metadata_id_cols:
            # For each item, get the list of feature values (e.g., list of genre_ids)
            sequences_for_meta_col = []
            for item_id in item_ids_list:
                item_meta = self.item_to_metadata_map.get(item_id, {}) # Get metadata for the item
                # Get specific feature list (e.g., genre_ids list for this item)
                # Default to empty list if item or specific metadata not found
                meta_feature_list = item_meta.get(meta_col_original, []) 
                sequences_for_meta_col.append(torch.tensor(meta_feature_list, dtype=torch.long))
            
            # Pad all sequences for the current metadata column to the same length
            padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences_for_meta_col, batch_first=True, padding_value=0)
            meta_feature_tensors_for_current_batch.append(padded_sequences)

        if not meta_feature_tensors_for_current_batch:
            return torch.empty(len(item_ids_list), 0, dtype=torch.long) # No metadata features processed

        # Stack along a new dimension if multiple metadata features, otherwise just use the single feature's tensor
        # This should result in shape (batch_size, num_meta_features, max_sequence_length_of_meta_feature)
        # or (batch_size, max_sequence_length_of_meta_feature) if only one meta feature type
        final_stacked_tensor = torch.stack(meta_feature_tensors_for_current_batch, dim=1) \
            if len(meta_feature_tensors_for_current_batch) > 1 else meta_feature_tensors_for_current_batch[0]
        
        return final_stacked_tensor


    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        end_idx = min(self.i + self.batch_size, self.dataset_len)
        
        if self.shuffle and self.dataset_len > 0:
            current_indices = self.indices[self.i:end_idx]
            batch = {key: tensor[current_indices] for key, tensor in self.data.items() if key in ['user_id', 'pos_item_id', 'pos_metadata_id']}
            # Fixed negatives would also be sliced here if not dynamic_neg_sampling
            if not self.dynamic_neg_sampling and 'neg_item_id' in self.data:
                 batch['neg_item_id'] = self.data['neg_item_id'][current_indices]
            if not self.dynamic_neg_sampling and 'neg_metadata_id' in self.data:
                 batch['neg_metadata_id'] = self.data['neg_metadata_id'][current_indices]
        else:
            batch = {key: tensor[self.i:end_idx] for key, tensor in self.data.items() if key in ['user_id', 'pos_item_id', 'pos_metadata_id']}
            if not self.dynamic_neg_sampling and 'neg_item_id' in self.data:
                 batch['neg_item_id'] = self.data['neg_item_id'][self.i:end_idx]
            if not self.dynamic_neg_sampling and 'neg_metadata_id' in self.data:
                 batch['neg_metadata_id'] = self.data['neg_metadata_id'][self.i:end_idx]

        if self.dynamic_neg_sampling:
            pos_item_ids = batch['pos_item_id'].tolist()
            current_batch_size = len(pos_item_ids)
            
            neg_item_ids_list = []
            for k in range(current_batch_size):
                pos_id = pos_item_ids[k]
                neg_id = np.random.randint(0, self.n_items)
                while neg_id == pos_id: # Ensure neg_id is different from pos_id
                    neg_id = np.random.randint(0, self.n_items)
                neg_item_ids_list.append(neg_id)
            
            batch['neg_item_id'] = torch.tensor(neg_item_ids_list, dtype=torch.long)

            if self.item_to_metadata_map and self.metadata_id_cols and 'pos_metadata_id' in batch: # Check if positive metadata exists to guide structure
                # ^ The 'pos_metadata_id' in batch check is a bit of a proxy to see if metadata is being used at all.
                # A more direct check would be if self.metadata_id_cols is not empty.
                neg_metadata_tensor = self._get_padded_metadata_tensor(neg_item_ids_list)
                if neg_metadata_tensor is not None:
                    batch['neg_metadata_id'] = neg_metadata_tensor
        
        self.i += self.batch_size
        
        return batch
