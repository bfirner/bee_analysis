"""
A network that begins with a sparse encoding, meant to be similar to Kenyon Cells in insects.
"""

import itertools
import torch
import torch.nn as nn


class SparseEncodingNet(nn.Module):
    """A convolution, then a modulation, then a n-way hash, thresholding, and prediction+feedback modulation."""

    def createConvolutions(self, width, depth, input_size, kc_repititions, activation=nn.Step):
        """Create 3x3 convolution layers of the given width and depth."""
        layers = []
        padding = 1
        stride = 2
        kernel_size = 3
        for d in range(depth):
            layers.append(nn.Conv2d(
                in_channels=input_size[0], out_channels=width,
                kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(activation())
            # Adjust the size for the next layer
            input_size = (
                width,
                int((input_size[1] + 2 * padding - kernel_size)/stride + 1),
                int((input_size[1] + 2 * padding - kernel_size)/stride + 1)
            )
            # TODO Initialize the bias to 0 and the weights
            with torch.no_grad():
                # Weights should be sparse, different, and in the range from 0 to 1.
                # TODO A percentage should have a single cell with weight around one, two cells with
                # weights 0.5, three with 0.3, etc. Something like that.
                # May be simpler to set the weight to a negative value so that all the weights must
                # be used to get a positive output and trigger the step activation function
                # For now, we'll just be using the orthogonal initialization without digging further
                # into this.
                torch.nn.init.orthogonal_(layers[-1].weight, gain=1)
                # TODO The bias? Set them all to the same value and then gradually lower if a layer
                # is overactive? Ideally, should each output be active half the time so that they
                # are all super-sensitive and then tuning happens in the KC-like layer? Should the
                # biases start extremely high and then gradually lower until 50% activation is
                # found?
        kc_selectors = nn.ModuleList()
        flatsize = input_size[0]*input_size[1]*input_size[2]
        for _ in kc_repititions:
            # Generate a random mapping by selecting a single input for each output of a matrix
            choices = shuffle(list(range(flatsize)))
            selector = torch.zeros([flatsize, flatsize])
        # Return the convolution layers and the output size
        return nn.Sequential(*block), input_size

    def forward(self, x):
        # First, run through the convolutions
        x = self.features(x)
        # Now, flatten and assign to positions in the KC bank with some masked selects
        x = x.flatten()
        # Now, assign the inputs to their outputs with a multiplication
        x = torch.zeros(x.size())
        for mask in self.kc_masks:
            x = x + torch.masked_select(x, mask)
        # TODO Apply modulation
        x = x * self.modulation_vector
        # Apply a threshold to only accept the top N outputs.
        # TODO Loop through with different thresholds until one is found that emits below a given %
        # of the outputs
        # TODO Evaluate the output and adjust the modulation_vector if they are "insufficient"

    def __init__(self, width, depth, input_size):
        """Initialize the sparse encoding network."""
        self.features, self.feature_size = self.createConvolutions(width, depth, input_size)
        pass

