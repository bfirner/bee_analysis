"""
A tiny network that serves as a way to undo normalization done during training.
Applies the function:
    f(x) = x * stddev + mean
This makes it easy to create a sequential network with this one following the DNN with normalized
output
"""

import torch
import torch.nn as nn


class Denormalizer(nn.Module):
    """Denormalization network"""

    def __init__(self, means, stddevs):
        super(Denormalizer, self).__init__()

        # These must be the same size and should be a single dimension
        assert means.size() == stddevs.size()
        assert means.dim() == 1

        # We could also represent this as a linear layer with weight=stddevs and bias=mean
        # That seems more complicated than is necessary
        self.means = nn.parameter.Parameter(means)
        self.stddevs = nn.parameter.Parameter(stddevs)

    def forward(self, x):
        # TODO FIXME with torch.nograd():
        return x * self.stddevs + self.means


class Normalizer(nn.Module):
    """Normalization network"""

    def __init__(self, means, stddevs):
        super(Normalizer, self).__init__()

        # These must be the same size and should be a single dimension
        assert means.size() == stddevs.size()
        assert means.dim() == 1

        # The normalization function is f(x) = (x-mean)/stddev = x*(1/stddev) - mean/stddev
        # We could also represent this as a linear layer with weight=1/stddev and bias=mean/stddev
        # Absent a reason to do that, we will be using the straightforward math, but will still
        # convert the division step into a multiplication to speed things up.
        self.means = nn.parameter.Parameter(means)
        self.inv_stddevs = nn.parameter.Parameter(1.0 / stddevs)

    def forward(self, x):
        # TODO FIXME with torch.nograd():
        return (x - self.means) * self.inv_stddevs
