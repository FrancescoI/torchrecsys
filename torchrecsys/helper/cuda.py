# -*- coding: utf-8 -*-

def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor