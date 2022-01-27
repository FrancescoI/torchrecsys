import sys
sys.path.append('C:/Users/imbrigliaf/Documents/torchrecsys/')

import pytest
import pandas as pd
from torchrecsys.dataset.dataset import CustomDataset
from torch.utils.data.dataloader import DataLoader


interactions = pd.DataFrame({'user': [0, 0, 1, 1, 2],
                             'item': [0, 1, 0, 2, 2],
                             'gender': [0, 1, 0, 1, 1],
                             'category': [0, 1, 0, 0, 0]})

my_data = CustomDataset(dataset=interactions,
                        user_id_col='user',
                        item_id_col='item',
                        metadata_id_col=['gender', 'category'])

my_data_loader = DataLoader(my_data, batch_size=2)

print(next(iter(my_data_loader)))