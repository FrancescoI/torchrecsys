# -*- coding: utf-8 -*-

import torch

class ScaledEmbedding(torch.nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Parameters:
    -----------
    num_embeddings: int
        Size of the dictionary of embeddings.
    embedding_dim: int
        The size of each embedding vector.
    padding_idx: int, optional
        If specified, the entries at `padding_idx` do not contribute to the gradient;
        therefore, the embedding vector at `padding_idx` is not updated during training,
        i.e. it remains as a fixed vector.
    max_norm: float, optional
        If given, each embedding vector with norm larger than `max_norm`
        is renormalized to have norm `max_norm`.
    norm_type: float, optional
        The p of the p-norm to compute for the `max_norm` option. Default `2`.
    scale_grad_by_freq: bool, optional
        If given, this will scale gradients by the inverse of frequency of the words
        in the mini-batch. Default `False`.
    sparse: bool, optional
        If `True`, gradient w.r.t. `weight` matrix will be a sparse tensor.
        See Notes for more details regarding sparse gradients. Default `False`.
    _weight: Tensor, optional
        If specified, the content of the given Tensor will be copied into the
        embedding layer's weight.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None): 
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq,
                         sparse, _weight) 

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
            
class ZeroEmbedding(torch.nn.Embedding):
    """
    Embedding layer that initialises its values to zero.
    Typically used for bias terms.

    Parameters:
    -----------
    num_embeddings: int
        Size of the dictionary of embeddings.
    embedding_dim: int
        The size of each embedding vector.
    padding_idx: int, optional
        If specified, the entries at `padding_idx` do not contribute to the gradient;
        therefore, the embedding vector at `padding_idx` is not updated during training,
        i.e. it remains as a fixed vector.
    max_norm: float, optional
        If given, each embedding vector with norm larger than `max_norm`
        is renormalized to have norm `max_norm`.
    norm_type: float, optional
        The p of the p-norm to compute for the `max_norm` option. Default `2`.
    scale_grad_by_freq: bool, optional
        If given, this will scale gradients by the inverse of frequency of the words
        in the mini-batch. Default `False`.
    sparse: bool, optional
        If `True`, gradient w.r.t. `weight` matrix will be a sparse tensor.
        See Notes for more details regarding sparse gradients. Default `False`.
    _weight: Tensor, optional
        If specified, the content of the given Tensor will be copied into the
        embedding layer's weight.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None): 
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq,
                         sparse, _weight) 

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)