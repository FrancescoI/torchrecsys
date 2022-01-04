import sys
sys.path.append('C:/Users/imbrigliaf/Documents/torchrecsys/')

import pytest
import pandas as pd
from torchrecsys.dataset.dataset import CustomDataset

interactions = pd.DataFrame({'user_id': [0, 0, 1, 1, 2],
                             'item_id': [0, 1, 0, 2, 2],
                             'gender_id': [0, 1, 0, 1, 1],
                             'category_id': [0, 1, 0, 0, 0]})

my_data = CustomDataset(dataset=interactions,
                        user_id_col='user_id',
                        item_id_col='item_id',
                        metadata_id_col=['gender_id', 'category_id'])

print(next(iter(my_data)))
