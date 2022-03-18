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
from torchrecsys.dataset.dataset import ProcessData
import pandas as pd

### Create random user-item interactions
interactions = pd.DataFrame({'user': np.random.choice(np.arange(3_000), size=100_000),
                             'item': np.random.choice(np.arange(1_000), size=100_000)})

data = ProcessData(dataset=interactions,
                   user_id_col='user',
                   item_id_col='item',
                   split_ratio=0.9)

my_data.write_data(path='/my_path')

model = TorchRecSys(path='/my_path',
                    net_type='linear',
                    use_cuda=False)

my_optimizer = Adam(model.parameters(), 
                    lr=4e-3,
                    weight_decay=1e-8)

model.fit(optimizer=my_optimizer, 
          epochs=30,
          batch_size=1_024)

model.evaluate(metrics=['loss', 'auc'])          
```