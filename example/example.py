# -*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/imbrigliaf/Documents/torchrecsys/')

from torchrecsys.model import *
from torchrecsys.dataset.dataset import *

my_data = MyDataset(brand='missoni')

my_data.fit(metadata=['saleLine', 'macro'])

model = TorchRecSys(dataset=my_data, 
                    n_factors=512, 
                    net_type='mlp', 
                    use_metadata=True,
                    use_cuda=False)

optimizer = torch.optim.Adam(model.net.parameters(), lr=4e-3, weight_decay=1e-6)

model.fit(optimizer=optimizer,
          batch_size=24_000,
          epochs=40,
          split_train_test=True,
          verbose=True)