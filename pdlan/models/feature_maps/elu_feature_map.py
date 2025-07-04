import torch
from torch.nn import Module

class ActivationFunctionFeatureMap(Module):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__()
        self.query_dims = query_dims
        self.activation_function = activation_function

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        return

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        return self.activation_function(x)

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner

elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)
