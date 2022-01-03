# -*- coding: utf-8 -*-

import torch

def hinge_loss(positive, negative):
    
    loss = torch.clamp(negative - positive + 1.0, 0.0)

    return loss.mean()