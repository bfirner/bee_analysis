"""
This model is useful to demonstrate how convnets operate.
It is specifically set up to detect the shapes created by the "CreateShapeData" tool.
"""

import itertools
import math
import torch
import torch.nn as nn

# Import the shapes from the shape creator to use as detection templates.
from datagen.CreateShapeData import all_shapes

def chunkifyShape(shape, stride):
    """
    Slice a shape into 3x3 chunks.
    Chunks will be sampled with interval determined by stride.
    nxn kernels overlap with strides < n.
    If the stride is too large then some shapes may not be properly sampled.

    The returned coordinates are in the output, not the input space. For example, a 5x5 shape split
    into chunks with a stride of 3 will have four output chunks in coordinates (0,0), (0,1), (1,0),
    and (1,1).

    Arguments:
        shapes (list[tensor]): List of tensors with shapes that should be detected. 
        stride          (int): Sampling stride.

    Returns:
        tuple (list[(int, int)], list[torch.tensor]): Coordinates and chunks
    """
    # Convert 2D tensors for 3D and process them both the same way.
    if shape.dim() == 2:
        shape = shape.expand(1, -1, -1)
    coordinates = []
    chunks = []
    # A 3x3 area has a 'receptive field' of 2 (meaning that position x,y will include pixels up to
    # x+2,y+2) so strides smaller than 3 will create chunks that overlap.
    # A large stride may cause us to miss the end of the shape.
    for xi, x in enumerate(range(0, shape.size(-1), stride)):
        for yi, y in enumerate(range(0, shape.size(-2), stride)):
            chunk = torch.zeros(shape.size(0), 3,3)
            # Indexing beyond the end of the tensor just returns up to the end of the tensor. Use
            # that to limit the area of the chunk being assigned into if the filter size and stride
            # do not really match the shape.
            chunk[:, :shape.size(-2)-y, :shape.size(-1)-x] = shape[:, y:y+3, x:x+3]
            # Don't bother with chunks that only have 0 values.
            if 0 < chunk.sum().item():
                chunks.append(chunk)
                coordinates.append((xi, yi))
    return (coordinates, chunks)


def chunkifyShapes(shapes, stride):
    """
    Slices features into 3x3 chunks. Returns the chunks in a list of chunk lists.
    Chunks will be sampled with interval determined by stride.
    nxn kernels overlap with strides < n.
    If the stride is too large then some shapes may not be properly sampled.

    Arguments:
        shapes (list[tensor]): List of tensors with shapes that should be detected. 
        stride          (int): Sampling stride.
    """
    return [chunkifyShape(shape, stride) for shape in shapes]


def createNextShapes(filter_coordinates, filter_channels, input_channels):
    """Create 3D shapes to detect the outputs of filters at the given coordinates.

    Arguments:
        filter_coordinates (list[list[tuple(x, y)]]): List of relative x,y coordinates of a
                                                filter with respect to other filters for
                                                this shape.
        filter_channels (list[int]): The input channels used by this filter.
        input_channels        (int): Number of input channels.
    Returns:
        torch.tensor: 3D tensor with 1s where a filter match would produce them.
    """
    next_shapes = []
    for shape_idx, coordinates in enumerate(filter_coordinates):

        # First find the dimensions of the next filter for this shape.
        max_x_coord = 0
        max_y_coord = 0
        for coordinate in coordinates:
            max_x_coord = max(max_x_coord, coordinate[0])
            max_y_coord = max(max_y_coord, coordinate[1])
        next_shape = torch.zeros(input_channels, max_y_coord+1, max_x_coord+1)

        # Mark the parts of the feature map that should have been activated if the shape was
        # present
        for chunk_idx, coordinate in enumerate(coordinates):
            channel = filter_channels[shape_idx][chunk_idx]
            next_shape[channel,coordinate[1],coordinate[0]] = 1
        next_shapes.append(next_shape)
    return next_shapes


def create3x3ConvLayer(in_channels, weights_and_biases, out_size):
    """
    Create a new 3x3 convolution + ReLU layer that detects the given features. Returns the
    layer, feature map output size, and the new features that will be present in the output
    feature map.

    Arguments:
        in_channels        (int): Number of channels of the input feature map or image.

        weights_and_biases (list[tuple(float, tensor)]): Weights and biases for each filter.
                                  First dimension of weights must match input channels.
        out_size                (tuple): Height and width of the current feature maps.
    Returns:
        tuple(nn.Sequential, tuple(int, int)): A tuple of the convolution layer and output size.
    """
    block = []

    ksize = 3
    stride = 3
    padding = 0
    block.append(nn.Conv2d(
        in_channels=in_channels, out_channels=len(weights_and_biases),
        kernel_size=ksize, stride=stride, padding=padding))
    out_size = (int((out_size[0] + 2 * padding - ksize)/stride + 1),
                int((out_size[1] + 2 * padding - ksize)/stride + 1))

    # Set the weights and biases
    for idx, (weight, bias) in enumerate(weights_and_biases):
        block[-1].bias[idx] = bias
        block[-1].weight[idx] = weight

    block.append(nn.ReLU())

    return nn.Sequential(*block), out_size

def createFilterForFeature(feature):
    """
    Create weights and a bias value for a 3x3 convolution for the given feature, as defined by
    feature.

    Arguments:
        feature (torch.tensor): 3D tensor of CxHxW coordinates of the shape. Coordinates
                                are relative to an arbitrary starting point and the height and
                                width dimensions should be less than 3. Use chunkifyShapes to
                                chunk larger shapes to the appropriate sizes.
    Returns:
        torch.tensor: Weights
        float       : Bias
    """
    # Create a filter that rejects anything that is not this shape. Assume that all
    # values in the image are either 0 or 1. Any location with a 0 in the chunk will
    # have a value of -filter_sum so the output of the filter will be 0 or less if
    # there is any 1 present where it is unexpected for this chunk.
    filter_sum = feature.sum().item()
    filter_weight = feature.clone() + ((feature-1.) * filter_sum)
    # If the filter encounters an exact match then the output will be filter_sum. The
    # bias should be set to -filter_sum + 1 so that output is supressed unless all of
    # the expected 1 values are encountered.
    filter_bias = -filter_sum + 1
    return filter_weight, filter_bias


class ShapeNet(nn.Module):
    """A network for detecting shapes from the CreateShapeData tool."""

    def __init__(self, in_dimensions, target_shapes = list(all_shapes.values())):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            target_shapes        (int): The shapes to detect.
        """
        super(ShapeNet, self).__init__()

        self.in_dimensions = in_dimensions
        self.out_classes = len(target_shapes)

        # The model will be a series of convolutions (with some supporting layers, such as
        # activation functions).
        self.model = nn.ModuleList()

        # Keep track of the size of the features maps at each step.
        out_size = in_dimensions[1:]

        # Initialize in a no_grad section to allow for some math to take place.
        with torch.no_grad():
            # Go through every shape, split them into 3x3 chunks, and create filters from the chunks
            # Repeat until every shape has been isolated to a to a single feature map. At that point
            # the shapes are separable.

            next_shapes = target_shapes
            shapes_separated = False

            input_channels = self.in_dimensions[0]

            # Keep making layers until the number of feature maps matches the number of input
            # shapes.
            while (not shapes_separated):
                # Now find the filters for the next layer and keep track of which channel they will
                # output into and their relative position to other filters for the same shape.
                weight_bias_tuples = []
                shape_output_channels = []
                filter_coordinates = []
                for shape in next_shapes:
                    # If the shape is larger than the 3x3 convolution receptive field it needs to be
                    # chunkified. Just send everything to the chunkifyShape function and let it sort
                    # them out.
                    chunk_coordinates, chunks = chunkifyShape(shape=shape, stride=3)
                    # Remember the relative position of each chunk with respect to the other chunks for
                    # this shape.
                    filter_coordinates.append(chunk_coordinates)

                    # Create a filter for each chunk.
                    # Need to keep track of which channels will be used by this shape's chunk detectors.
                    output_channels = []
                    for chunk in chunks:
                        weight_bias_tuple = createFilterForFeature(chunk)
                        # In the event that this filter already exists it would be wasteful to repeat it.
                        # Otherwise insert this (weight, bias) tuple for filter creation.
                        # Either way record the channel that corresponds to detection of this chunk.
                        prev_match = [(weight_bias_tuple[0] == weights).all().item() for weights, _ in weight_bias_tuples]
                        if True in prev_match:
                            output_channels.append(prev_match.index(True))
                        else:
                            output_channels.append(len(weight_bias_tuples))
                            weight_bias_tuples.append(weight_bias_tuple)
                    # Record the output channels that correspond to all features of this shape.
                    shape_output_channels.append(output_channels)

                # Prepare the shapes for the next iteration
                next_shapes = createNextShapes(
                    filter_coordinates, shape_output_channels, len(weight_bias_tuples))

                # Create and append the new convolution layer
                layer, out_size = create3x3ConvLayer(in_channels=input_channels,
                    weights_and_biases=weight_bias_tuples, out_size=out_size)
                self.model.append(layer)
                # Update the number of channels for the next layer
                input_channels = len(weight_bias_tuples)
                # Filtering is complete if there is one output channel per shape.
                total_used_channels = sum([len(channels) for channels in shape_output_channels])
                shapes_separated = len(target_shapes) == total_used_channels

            # Run max pooling on the output channels.
            # The output is the detector.
            self.classifier = nn.Sequential(
                nn.MaxPool2d(kernel_size=(out_size[0], out_size[1])),
                nn.Flatten(),
            )


    #TODO AMP
    #@autocast()
    def forward(self, x):
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        for layer in self.model:
            x = layer(x)

        x = self.classifier(x)
        return x
