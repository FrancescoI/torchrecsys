# TorchRecSys

TorchRecSys is a Python implementation of several Collaborative Filtering and Sequence Models for recommender systems, using PyTorch as backend (with single GPU training mode available).

Side information (alias: product metadata) can be used too. 

Available models for Collaborative Filterings:
* A simplified version of LightFM by Kula (called "Linear")
* MLP
* FM
* NeuCF and EASE (yet to come) 

Available models for Sequence models:
* LSTM (yet to come)

For more details, see the [Documentation]().

## Installation
Install from `pip`:
```
pip install torchrecsys
```


## Quickstart
Fitting a collaborative filtering (eg: MLP) is very easy:
```python
from torchrecsys.model import *
from torch.optim import Adam
import pandas as pd

### Create random user-item interactions
interactions = pd.DataFrame({'user': np.random.choice(np.arange(3_000), size=100_000),
                             'item': np.random.choice(np.arange(1_000), size=100_000)})

data_loader = CustomDataLoader(dataset=interactions,
                               user_id_col='user',
                               item_id_col='item',
                               use_metadata=False,
                               split_ratio=0.9)

model = TorchRecSys(dataloader=data_loader,
                    net_type='mlp',
                    use_metadata=False,
                    use_cuda=False)

my_optimizer = Adam(model.parameters(), lr=4e-3)

model.fit(dataloader=data_loader, 
          optimizer=my_optimizer, 
          epochs=30,
          batch_size=128)

model.evaluate(metrics=['loss', 'auc'])          
```