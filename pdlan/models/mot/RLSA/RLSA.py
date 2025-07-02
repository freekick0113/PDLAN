#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the causally masked linear attention as a recurrent model."""

import torch
from torch import nn
from torch.nn import Module

from ...attention_registry import RecurrentAttentionRegistry, Optional, Int, \
    Callable, EventDispatcherInstance
from ...events import EventDispatcher
from ...feature_maps import elu_feature_map
from ._utils import check_state


class RecurrentLinearSelfAttention(Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self,channels, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(RecurrentLinearSelfAttention, self).__init__()
        self.feature_map = (
            feature_map(channels) if feature_map else
            elu_feature_map(channels)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)

    def forward(self,feature, state=None, memory=None):
        feature = feature.flatten(2).transpose(1, 2)
        if torch.cuda.is_available():
            feature = feature.to('cuda') 
        query = self.w_q(feature)
        key = self.w_k(feature)
        value = self.w_v(feature)

        # Normalize state/memory
        state = check_state(state, memory)

        # If this is a new sequence reinitialize the feature map
        if state is None:
            self.feature_map.new_feature_map(query.device)

        # Apply the feature map to the query and key
        Q = self.feature_map.forward_queries(query) #f(Q)
        K = self.feature_map.forward_keys(key) #f(K)

        B, N, C=Q.shape
        _, _, M = value.shape

        # Extract the memory or initialize it
        if state is None:
            Ri = query.new_zeros((B, N, C, M))
            Si = query.new_zeros((B, N, C))
        else:
            Ri, Si = state

        # Ensure the batch size did not change
        if len(Ri) != B:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        if K.grad_fn is not None or value.grad_fn is not None:
            Si = Si + K #sum_f(K)
            Ri = Ri + torch.einsum("bnc,bnm->bncm", K, value) 

        else:
            Si += K
            Ri += torch.einsum("bnc,bnm->bncm", K, value)

        # Compute the output
        Z = 1. / (torch.einsum("bnc,bnc->bn", Q, Si) + self.eps) 
        RLSA = torch.einsum("bnc,bncm,bn->bnm", Q, Ri, Z) 

        return RLSA, [Ri, Si]


# Register the attention implementation so that it becomes available in our
# builders
RecurrentAttentionRegistry.register(
    "linear", RecurrentLinearSelfAttention,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
RecurrentAttentionRegistry.register(
    "causal-linear", RecurrentLinearSelfAttention,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
