import unittest
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam

from torchrecsys.model import TorchRecSys
from torchrecsys.dataset.dataset import ProcessData, FastDataLoader
from torchrecsys.collaborative.mlp import MLP # For direct MLP testing

# Global dummy data for consistency
N_USERS = 100
N_ITEMS = 50
N_INTERACTIONS = 1000
N_METADATA_CATEGORIES = 5 # Number of categories for a single metadata feature

def get_dummy_interactions_data(with_metadata=False):
    users = np.random.choice(np.arange(N_USERS), size=N_INTERACTIONS)
    items = np.random.choice(np.arange(N_ITEMS), size=N_INTERACTIONS)
    df = pd.DataFrame({'user_id': users, 'item_id': items})
    if with_metadata:
        # Adding a single metadata column with lists of category IDs
        # Ensuring metadata values are lists of integers (simulating processed metadata)
        df['category_ids'] = [list(np.random.choice(np.arange(N_METADATA_CATEGORIES), size=np.random.randint(1, 4))) for _ in range(N_INTERACTIONS)]
    return df

class TestTorchRecSysFeatures(unittest.TestCase):

    def setUp(self):
        self.dummy_interactions = get_dummy_interactions_data(with_metadata=False)
        self.dummy_interactions_with_meta = get_dummy_interactions_data(with_metadata=True)
        self.user_id_col = 'user_id'
        self.item_id_col = 'item_id'
        self.metadata_cols = ['category_ids'] # Matches the key in get_dummy_interactions_data

    def _get_model_instance(self, dataset, **kwargs):
        base_params = {
            'dataset': dataset,
            'user_id_col': self.user_id_col,
            'item_id_col': self.item_id_col,
            'n_factors': 16, # Smaller factors for faster tests
            'use_cuda': False, # Default to False for most tests
            'net_type': 'linear', # Default net_type
        }
        base_params.update(kwargs)
        if 'metadata_id_col' in base_params and base_params['metadata_id_col'] is None:
            del base_params['metadata_id_col'] # Avoid passing None if not used by test

        return TorchRecSys(**base_params)

    # 1. Test Data Processing and Dynamic Negative Sampling
    def test_process_data_static_neg_sampling(self):
        data_processor = ProcessData(dataset=self.dummy_interactions,
                                     user_id_col=self.user_id_col,
                                     item_id_col=self.item_id_col,
                                     dynamic_neg_sampling=False)
        data_processor.prepare_data()
        self.assertIn('neg_item_id', data_processor.train_data)
        self.assertTrue(data_processor.train_data['neg_item_id'].numel() > 0)

    def test_process_data_static_neg_sampling_with_meta(self):
        data_processor = ProcessData(dataset=self.dummy_interactions_with_meta,
                                     user_id_col=self.user_id_col,
                                     item_id_col=self.item_id_col,
                                     metadata_id_col=self.metadata_cols,
                                     dynamic_neg_sampling=False)
        data_processor.prepare_data()
        self.assertIn('neg_item_id', data_processor.train_data)
        self.assertTrue(data_processor.train_data['neg_item_id'].numel() > 0)
        self.assertIn('neg_metadata_id', data_processor.train_data)
        self.assertTrue(data_processor.train_data['neg_metadata_id'].numel() > 0)


    def test_process_data_dynamic_neg_sampling(self):
        data_processor = ProcessData(dataset=self.dummy_interactions,
                                     user_id_col=self.user_id_col,
                                     item_id_col=self.item_id_col,
                                     dynamic_neg_sampling=True)
        data_processor.prepare_data()
        self.assertNotIn('neg_item_id', data_processor.train_data)

    def test_process_data_dynamic_neg_sampling_with_meta(self):
        data_processor = ProcessData(dataset=self.dummy_interactions_with_meta,
                                     user_id_col=self.user_id_col,
                                     item_id_col=self.item_id_col,
                                     metadata_id_col=self.metadata_cols,
                                     dynamic_neg_sampling=True)
        data_processor.prepare_data()
        self.assertNotIn('neg_item_id', data_processor.train_data)
        self.assertNotIn('neg_metadata_id', data_processor.train_data)


    def test_dataloader_dynamic_sampling_batch(self):
        model = self._get_model_instance(dataset=self.dummy_interactions, dynamic_neg_sampling=True)
        # Access the internal FastDataLoader for training
        train_loader = FastDataLoader(data=model.data_processor.train_data,
                                      batch_size=32,
                                      shuffle=False, # Keep shuffle false for easier testing if needed
                                      dynamic_neg_sampling=model.dynamic_neg_sampling,
                                      n_items=model.n_items,
                                      item_to_metadata_map=model.data_processor.item_to_metadata_map,
                                      metadata_id_cols=model.metadata_name)
        
        batch = next(iter(train_loader))
        self.assertIn('neg_item_id', batch)
        self.assertEqual(batch['pos_item_id'].shape, batch['neg_item_id'].shape)
        
        for pos_id, neg_id in zip(batch['pos_item_id'], batch['neg_item_id']):
            self.assertNotEqual(pos_id.item(), neg_id.item())

    def test_dataloader_dynamic_sampling_batch_with_meta(self):
        model = self._get_model_instance(dataset=self.dummy_interactions_with_meta, 
                                         metadata_id_col=self.metadata_cols,
                                         dynamic_neg_sampling=True)
        
        train_loader = FastDataLoader(data=model.data_processor.train_data,
                                      batch_size=32,
                                      shuffle=False,
                                      dynamic_neg_sampling=model.dynamic_neg_sampling,
                                      n_items=model.n_items,
                                      item_to_metadata_map=model.data_processor.item_to_metadata_map,
                                      metadata_id_cols=model.metadata_name)
        
        batch = next(iter(train_loader))
        self.assertIn('neg_item_id', batch)
        self.assertIn('neg_metadata_id', batch)
        self.assertEqual(batch['pos_item_id'].shape[0], batch['neg_item_id'].shape[0])
        self.assertEqual(batch['pos_metadata_id'].shape[0], batch['neg_metadata_id'].shape[0])
        # Basic shape check for metadata (batch_size, num_meta_features_stacked, max_meta_len)
        # or (batch_size, max_meta_len) if single meta feature
        self.assertTrue(len(batch['neg_metadata_id'].shape) == 2 or len(batch['neg_metadata_id'].shape) == 3)


    # 2. Test Mixed-Precision Training (AMP)
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping AMP test.")
    def test_amp_training_run(self):
        model = self._get_model_instance(dataset=self.dummy_interactions, use_amp=True, use_cuda=True)
        optimizer = Adam(model.parameters(), lr=1e-3)
        try:
            model.fit(optimizer, epochs=1, batch_size=32)
        except Exception as e:
            self.fail(f"AMP training run failed with exception: {e}")

    # 3. Test MLP Configuration (Direct MLP Instantiation)
    def test_mlp_custom_hidden_layers(self):
        custom_layers = [64, 32]
        # For direct MLP test, we need n_users, n_items, n_metadata, n_factors
        # Get these from a ProcessData instance as TorchRecSys would
        pd_for_counts = ProcessData(self.dummy_interactions, self.user_id_col, self.item_id_col)

        mlp_model = MLP(n_users=pd_for_counts.num_users, 
                        n_items=pd_for_counts.num_items, 
                        n_metadata={}, # No metadata for this specific test of hidden layers
                        n_factors=16, 
                        use_metadata=False,
                        hidden_layers=custom_layers)
        
        self.assertEqual(len(mlp_model.fcs), len(custom_layers))
        self.assertEqual(mlp_model.fcs[0].out_features, custom_layers[0])
        self.assertEqual(mlp_model.fcs[1].out_features, custom_layers[1])
        self.assertEqual(mlp_model.output_layer.in_features, custom_layers[-1])

        # Create a dummy batch for forward pass
        # input_shape for MLP is (n_factors * 2) when no metadata
        dummy_batch_data = torch.randn(32, 16 * 2) # batch_size, input_shape
        
        # The MLP model's forward expects a dict, but for this direct test, we can adapt
        # or call a simplified forward if MLP has one.
        # For now, let's test the layer structure. A forward pass test requires more setup
        # to match the expected input dictionary structure of user_id, item_id etc.
        # This test focuses on layer creation.

    def test_mlp_batch_norm_toggle(self):
        pd_for_counts = ProcessData(self.dummy_interactions, self.user_id_col, self.item_id_col)
        
        # With Batch Norm
        mlp_bn_true = MLP(n_users=pd_for_counts.num_users, n_items=pd_for_counts.num_items, 
                          n_metadata={}, n_factors=16, use_metadata=False, use_batch_norm=True)
        self.assertTrue(hasattr(mlp_bn_true, 'bns'))
        self.assertEqual(len(mlp_bn_true.bns), len(mlp_bn_true.hidden_layers))

        # Without Batch Norm
        mlp_bn_false = MLP(n_users=pd_for_counts.num_users, n_items=pd_for_counts.num_items,
                           n_metadata={}, n_factors=16, use_metadata=False, use_batch_norm=False)
        self.assertFalse(hasattr(mlp_bn_false, 'bns') or (hasattr(mlp_bn_false, 'bns') and len(mlp_bn_false.bns) == 0) )


    # 4. Test Batched Prediction
    def test_batched_prediction(self):
        # Use a slightly larger N_ITEMS for this test to make batching meaningful
        local_n_items = 20 
        users = np.random.choice(np.arange(10), size=50)
        items = np.random.choice(np.arange(local_n_items), size=50)
        local_df = pd.DataFrame({'user_id': users, 'item_id': items})
        
        model = self._get_model_instance(dataset=local_df) # n_items will be derived by ProcessData
        
        preds = model.predict(user_id=0, top_k=5, prediction_batch_size=7)
        self.assertEqual(preds.shape[0], 5)
        self.assertTrue(all(idx < model.n_items for idx in preds))


    def test_batched_prediction_consistency(self):
        local_n_items = 25
        users = np.random.choice(np.arange(10), size=60)
        items = np.random.choice(np.arange(local_n_items), size=60)
        local_df = pd.DataFrame({'user_id': users, 'item_id': items})

        model = self._get_model_instance(dataset=local_df)

        preds_batched = model.predict(user_id=0, top_k=5, prediction_batch_size=7)
        preds_full = model.predict(user_id=0, top_k=5, prediction_batch_size=local_n_items + 1) # Effectively no item batching

        self.assertTrue(torch.equal(preds_batched, preds_full))
        self.assertEqual(preds_batched.shape[0], 5)


    # 5. Test Profiling Run
    def test_profiling_run(self):
        model = self._get_model_instance(dataset=self.dummy_interactions)
        optimizer = Adam(model.parameters(), lr=1e-3)
        try:
            # Redirect stdout to check for profiler output if necessary, but for now, just run
            model.fit(optimizer, epochs=1, batch_size=32, profile_epochs=1)
        except Exception as e:
            self.fail(f"Profiling run failed with exception: {e}")


if __name__ == '__main__':
    unittest.main()
