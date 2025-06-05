# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def get_negative_batch(batch, n_items, item_to_metadata_dict):

    neg_item_id = np.random.randint(0, n_items-1, len(batch))
    
    if item_to_metadata_dict:
        neg_metadata_id = [item_to_metadata_dict[item] for item in neg_item_id]    
    else:
        neg_metadata_id = None
    
    neg_batch = pd.DataFrame({
        'user_id': batch['user_id'],
        'item_id': neg_item_id,
        'metadata': neg_metadata_id
    })
            
    return neg_batch
