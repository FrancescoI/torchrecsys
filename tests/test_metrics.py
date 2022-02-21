import sys
sys.path.append('C:/Users/imbrigliaf/Documents/torchrecsys/')

import pytest
import numpy as np
from torchrecsys.evaluate.metrics import *


def test_hit_rate():

    y_hat = torch.from_numpy(np.array([[0, 1, -1], 
                                       [1, 2, -1], 
                                       [1, 2, 3],
                                       [0, 1, -1]]))

    y_pred = torch.from_numpy(np.array([[2, 3],
                                        [0, 2],
                                        [1, 2],
                                        [0, 3]]))

    metrics = Metrics()

    true = metrics.hit_rate(y_hat=y_hat, y_pred=y_pred)
    
    #expected = np.array([0, 1, 1, 1]).sum() / len(y_pred)
    expected = 3/4
    
    assert true == expected