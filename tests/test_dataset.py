import sys
sys.path.append('C:/Users/imbrigliaf/Documents/torchrecsys/')

import pytest
import pandas as pd
from torchrecsys.dataset.dataset import CustomDataset
from torch.utils.data.dataloader import DataLoader
import torch


interactions = pd.DataFrame({'user': [0, 0, 1, 1, 2],
                             'item': [0, 1, 0, 2, 2],
                             'gender': [0, 1, 0, 1, 1],
                             'category': [0, 1, 0, 0, 0]})

my_data = CustomDataset(dataset=interactions,
                        user_id_col='user',
                        item_id_col='item',
                        metadata_id_col=['gender', 'category'])


def test_dataloader_attributes():

    my_data_loader = DataLoader(my_data, batch_size=3, shuffle=False)

    assert my_data_loader.dataset.num_items == 3
    assert my_data_loader.dataset.num_users == 3
    assert len(my_data_loader.dataset.metadata_id) == 2


def test_batch_size():

    my_data_loader = DataLoader(my_data, batch_size=2, shuffle=False)

    batch = next(iter(my_data_loader))
    
    assert batch['user_id'].size()[0] == 2 


def test_loader():

    my_data_loader = DataLoader(my_data, batch_size=2, shuffle=False)
    
    batch = next(iter(my_data_loader))  

    assert torch.equal(batch['user_id'], torch.tensor([0, 0]))
    assert torch.equal(batch['positive_item_id']['item_id'], torch.tensor([0, 1]))
    
    for key in batch['positive_item_id']['metadata_id'].keys():
        assert torch.equal(batch['positive_item_id']['metadata_id'][key], torch.tensor([0, 1]))

