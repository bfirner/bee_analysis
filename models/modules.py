"""
Modules to undo normalization done during training.
Applies the function:
    f(x) = x * stddev + mean
This makes it easy to create a sequential network with this one following the DNN with normalized
output.
Contains additional useful modules.
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
        self.means = nn.parameter.Parameter(means, requires_grad=False)
        self.stddevs = nn.parameter.Parameter(stddevs, requires_grad=False)

    def forward(self, x):
        # Forward without changing the means and stddevs
        return x * self.stddevs.detach() + self.means.detach()


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
        self.means = nn.parameter.Parameter(means, requires_grad=False)
        self.inv_stddevs = nn.parameter.Parameter(1.0 / stddevs, requires_grad=False)

    def forward(self, x):
        # TODO FIXME with torch.nograd():
        return (x - self.means.detach()) * self.inv_stddevs.detach()


class PresolvedConv2d(nn.Module):
    """A 2d convolution layer with some parameters presolved."""

    def __init__(self, in_channels, out_channels, presolved_weights, presolved_bias, kernel_size, stride=1,
            padding=0, dilation=1, groups=1):
        super(PresolvedConv2d, self).__init__()

        # The weights and bias must be the same size in their first dimension
        assert presolved_weights.size(0) == presolved_bias.size(0)

        # Weights should have size: [out_channels, in_channels, height, width]
        solved_channels = presolved_weights.size(0)

        self.solved_weight = nn.parameter.Parameter(data=presolved_weights, requires_grad=False)
        self.solved_bias = nn.parameter.Parameter(data=presolved_bias, requires_grad=False)

        self.conv_arguments = {
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups
        }

        self.learned_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels-solved_channels, kernel_size=kernel_size, **self.conv_arguments)

    def forward(self, x):
        """Forward through the convs."""
        presolved = torch.nn.functional.conv2d(input=x, weight=self.solved_weight.data, bias=self.solved_bias.data, **self.conv_arguments)
        learned = self.learned_conv(x)

        return torch.cat((presolved, learned), dim=1)


class PresolvedLinear(nn.Module):
    """A linear layer with some parameters presolved."""

    def __init__(self, in_features, out_features, presolved_weights, presolved_bias):
        super(PresolvedLinear, self).__init__()

        # The weights and bias must be the same size in their first dimension
        assert presolved_weights.size(0) == presolved_bias.size(0)

        # Weights should have size: [out_features, in_features]
        solved_channels = presolved_weights.size(0)

        self.solved_weight = nn.parameter.Parameter(data=presolved_weights, requires_grad=False)
        self.solved_bias = nn.parameter.Parameter(data=presolved_bias, requires_grad=False)

        self.learned_linear = torch.nn.Linear(in_features=in_features, out_features=out_features-solved_channels)

    def forward(self, x):
        """Forward through the linear layers."""
        presolved = torch.nn.functional.linear(x, weight=self.solved_weight, bias=self.solved_bias)
        learned = self.learned_linear(x)

        return torch.cat((presolved, learned), dim=1)


class MaxThresholding(nn.Module):
    """A module that applies a threshold based upon the maximum value."""

    def __init__(self, alpha=0.9):
        """Anything less than 0.9 * max will be converted to 0. Outputs will be (input - alpha*max)"""
        super(MaxThresholding, self).__init__()
        self.alpha = 0.9
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        """Outputs will be 0 or (input - alpha*max)."""
        # Assume that the first dimension is a batch dimension.
        threshold = torch.max(x.flatten(1), dim=1)[0].view(x.size(0), *((x.dim()-1)*[1])).expand(x.size())
        return self.activation(torch.sub(x, threshold, alpha=self.alpha))

