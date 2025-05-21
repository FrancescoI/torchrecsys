# TorchRecSys

TorchRecSys is a Python implementation of several Collaborative Filtering and Sequence Models for recommender systems, using PyTorch as backend. It features an optimized data loading pipeline for faster training and reduced memory usage, along with single GPU training mode.

Side information (product metadata) can be easily integrated.

**Available models for Collaborative Filtering:**
*   **Linear:** A simplified version of LightFM by Kula.
*   **MLP:** Multi-Layer Perceptron with configurable hidden layers and optional Batch Normalization.
*   **FM:** Factorization Machines.
*   NeuCF and EASE (yet to come)

**Available models for Sequence models:**
*   LSTM (yet to come)

**Key Features:**
*   Optimized data processing: Data is pre-processed and tensorized in memory to minimize I/O overhead.
*   Dynamic Negative Sampling: Option for on-the-fly negative sampling during training (`dynamic_neg_sampling=True`).
*   Automatic Mixed Precision (AMP): Support for AMP training on CUDA GPUs (`use_amp=True`) for potentially faster training and reduced memory footprint.
*   Sparse Embeddings: Utilizes `torch.nn.Embedding(sparse=True)` for potentially more efficient optimizer updates with sparse gradients.
*   Configurable MLP: Define custom hidden layer architectures for the MLP model (e.g., `hidden_layers=[512, 256, 128]`).
*   Optional Batch Normalization: MLP model includes Batch Normalization by default (`use_batch_norm=True`), which can be disabled.
*   Built-in Profiler: Use `profile_epochs=1` in `model.fit()` to print a performance summary of the first training epoch.
*   Batched Predictions: Efficiently predicts scores for a large number of items by processing them in batches, reducing peak memory usage.

For more details, see the [Documentation]().

## Installation
Install from `pip`:
```
pip install torchrecsys
```

## Quickstart
Fitting a collaborative filtering model (e.g., MLP) is straightforward:

```python
import torch
import pandas as pd
import numpy as np
from torchrecsys.model import TorchRecSys
from torch.optim import Adam

# Create random user-item interactions
interactions_df = pd.DataFrame({
    'user': np.random.choice(np.arange(3_000), size=100_000),
    'item': np.random.choice(np.arange(1_000), size=100_000),
    # 'product_category': np.random.choice(np.arange(100), size=100_000) # Example metadata column
})

# Instantiate the model
# The data processing is now handled internally by TorchRecSys
model = TorchRecSys(dataset=interactions_df,
                    user_id_col='user',
                    item_id_col='item',
                    net_type='mlp',  # Using MLP to demonstrate its specific params
                    # metadata_id_col=['product_category'], # Uncomment if using metadata
                    # hidden_layers=[512, 256], # Optional: Configure MLP hidden layers
                    # dynamic_neg_sampling=True, # Optional: Enable dynamic negative sampling
                    use_cuda=False, # Set to True if a CUDA-enabled GPU is available
                    # use_amp=True, # Optional: Enable AMP if use_cuda=True
                    )

# Define optimizer
my_optimizer = Adam(model.parameters(), 
                    lr=1e-3, # Adjusted learning rate
                    weight_decay=1e-5) # Adjusted weight decay

# Fit the model
model.fit(optimizer=my_optimizer, 
          epochs=5, # Reduced epochs for quick example
          batch_size=1_024,
          profile_epochs=1) # Optional: Profile the first epoch

# Evaluate the model
model.evaluate(batch_size=1_024, eval_metrics=['loss', 'auc'])

# Get top K predictions for a user
user_id_to_predict = 0
top_k_items = model.predict(user_id=user_id_to_predict, top_k=10)
print(f"Top 10 item recommendations for user {user_id_to_predict}: {top_k_items.tolist()}")
```