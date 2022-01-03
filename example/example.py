# -*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/imbrigliaf/Documents/torchrecsys/')

from torchrecsys.model import *
from torchrecsys.dataset.dataset import *

missoni = MyDataset(brand='missoni')

missoni.fit(metadata=['saleLine', 'macro'])

model = TorchRecSys(dataset=missoni, 
                    n_factors=80, 
                    net_type='lightfm', 
                    use_metadata=True,
                    use_cuda=False)

optimizer = torch.optim.Adam(model.net.parameters(), lr=3e-3)

model.fit(optimizer=optimizer,
          batch_size=10_240,
          epochs=20,
          split_train_test=True,
          verbose=True)