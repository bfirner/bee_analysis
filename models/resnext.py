"""
The ResNext architecture is, as its name implies, an improvement to the original resnet
architecture made after the success of "inception" modules in neural networks. The
'split-transform-merge' technique from inception modules is brought to the residual network.
The advantage of this approach is that inputs are split into multiple simple pathways through a
module to keep total complexity low (compared to a fully expressive pathway) but the variety of the
different pathways chosen by the network architect still allows for high expressivity. For example,
if an input had 256 channels and was sent through a 5x5 convolution that produced a 256 channel
output some of those 5x5 kernels may only be as utilized as a 3x3 convolution, or may only use a
subset of the input channels. By splitting the input into smaller groups those same functions can be
realized without the cost of a full convolution over the entire space. The key argument of this
approach is that increasing the cardinality (the size of the set of transformations) is more
effective than making deeper or wider networks, especially since those approaches have diminishing
returns and increasing hardware costs.
See: https://arxiv.org/pdf/1611.05431
"""

import itertools
import math
import torch
import torch.nn as nn


class ResNext50(nn.Module):
    """ResNet50."""

    def createResLayer(self, i, out_size=None):
        """
        Each convolution is followed by BatchNorm2d and then by an activation function.
        Biases are initialized to 0 and weights are a zero-mean Gaussian with standard deviation
        sqrt(2/n), where n is the number of connections (kernels size squared * input channels)
        Arguments:
            i          (int): Current layer index.
            out_size (tuple): Height and width of the current feature maps.
        Returns:
            tuple(nn.Sequential, tuple(int, int)): A tuple of the convolution layer and output size.
        """
        block = []
        for j in range(3):
            groups = 1
            stride = 1
            ksize = 1
            padding = 0
            if 0 == j:
                input_size = self.channels[i][-1]
                out_channels = self.channels[i+1][0]
                # Decreasing the feature size only occurs in the first convolution.
                stride = self.strides[i]
            elif 1 == j:
                input_size = self.channels[i+1][0]
                out_channels = self.channels[i+1][1]
                # ReLU after the batch norm of the previous conv layer
                block.append(nn.ReLU())
                # The grouping is applied in the middle layer
                groups = self.groups[i]
                ksize = self.kernels[i]
                padding = self.padding[i]
            elif 2 == j:
                input_size = self.channels[i+1][1]
                out_channels = self.channels[i+1][2]
                # ReLU after the batch norm of the previous conv layer
                block.append(nn.ReLU())

            block.append(nn.Conv2d(
                in_channels=input_size, out_channels=out_channels, groups=groups,
                kernel_size=ksize, stride=stride, padding=padding))
            block[-1].bias.fill_(0)
            block[-1].weight.fill_(math.sqrt(input_size * self.kernels[i]**2))
            if out_size is not None:
                out_size = (int((out_size[0] + 2 * padding - ksize)/stride + 1),
                            int((out_size[1] + 2 * padding - ksize)/stride + 1))
            # Batch norm comes before the activation layer
            block.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*block), out_size

    def createLinearLayer(self, num_inputs, num_outputs):
        """
        Arguments:
            num_inputs  (int): Number of inputs to the linear layer.
            num_outputs (int): Number of outputs to the linear layer.
        Returns:
            nn.Sequential: The linear layer.
        """
        layer = nn.Sequential(
            nn.Linear(
                in_features=num_inputs, out_features=num_outputs),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        layer[0].bias.fill_(1.)
        return layer

    def initializeSizes(self):
        """
        Override this function to implement a different kind of residual network.
        """
        block1_channels = [128, 128, 256]
        block2_channels = [256, 256, 512]
        block3_channels = [512, 512, 1024]
        block4_channels = [1024, 1024, 2048]
        self.channels = ([self.in_dimensions[0]], [64], *([block1_channels] * 3), *([block2_channels] * 4),
                *([block3_channels] * 6), *([block4_channels] * 3))
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (7, *([3] * (3+4+6+3)))
        self.strides = (2, *[2, 1, 1], *[2, 1, 1, 1], *[2, 1, 1, 1, 1, 1], *[2, 1, 1])
        self.padding = (3, *([1] * (3+4+6+3)))
        self.groups = (1, *([32] * (3+4+6+3)))
        # The first 7x7 kernel is followed by a 3x3 pooling layer, so the 7 needs to be
        # adjusted to a 9.
        self.vis_mask_sizes = (9, *([3]*(3+4+6+3)))
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))

    def createInternalResLayers(self, out_size):
        """
        Override this function to implement a different kind of residual network.

        Arguments:
            out_size (list[int]): The width and height of the current feature maps.
        """
        # Now the residual layers
        for i in range(1, len(self.kernels)):
            # If the number of channels increases then project to a higher number of channels
            # Also project if the feature map size changes.
            # Otherwise just use an identity.
            if 1 < self.strides[i] or self.channels[i][-1] < self.channels[i+1][-1]:
                projection = nn.Conv2d(in_channels=self.channels[i][-1],
                    out_channels=self.channels[i+1][-1], stride=self.strides[i], padding=0, kernel_size=1)
                self.shortcut_projections.append(projection)
            else:
                self.shortcut_projections.append(nn.Identity())
            layer, out_size = self.createResLayer(i, out_size)

            # Dropout was not in the original paper, but we can provide and option for it.
            # TODO This should probably happen after the addition at the end of the shortcut though.
            if self.use_dropout:
                block.append(nn.Dropout2d(p=0.2))

            self.output_sizes.append(out_size)
            self.model.append(layer)
        return out_size

    def createVisMaskLayers(self, output_sizes):
        """
        Arguments:
            output_sizes (list[(int, int)]): Feature map dimensions.
        """
        self.vis_layers = nn.ModuleList()
        for i in range(len(self.vis_mask_sizes)):
            stride = self.strides[i]
            # The max pooling in the beginning has a stride of 2.
            if 0 == i:
                stride = stride * 2
            # The default output size of the ConvTranspose2d is the minimum dimension of the source
            # but the source could have had a larger dimension.
            # For example, if the dimension at layer i+1 is n with a stride of x then the source
            # could have had a dimension as small as n*x-(x-1) or as large as n*x.
            # To resolve this ambiguity we need to add in specific padding.
            max_source = (
                stride * output_sizes[i+1][0], stride * output_sizes[i+1][1])
            min_source = (
                max_source[0] - (stride - 1), max_source[1] - (stride - 1))
            output_padding = (
                output_sizes[i][0] - min_source[0], output_sizes[i][1] - min_source[1])
            # This awkward code is a quick fix for the output padding calculation being overly
            # large.
            if stride <= output_padding[0]:
                output_padding = (output_padding[0] - stride, output_padding[1])
            if stride <= output_padding[1]:
                output_padding = (output_padding[0], output_padding[1] - stride)
            padding = self.vis_mask_sizes[i] // 2
            self.vis_layers.append(nn.ConvTranspose2d(
                in_channels=1, out_channels=1, kernel_size=self.vis_mask_sizes[i],
                stride=stride, padding=padding, output_padding=output_padding, bias=False))
            self.vis_layers[-1].weight.requires_grad_(False)
            self.vis_layers[-1].weight.fill_(1.)

    def __init__(self, in_dimensions, out_classes, expanded_linear=False, use_dropout=False):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
            expanded_linear     (bool): True to expand the linear layers from the initial paper.
                                        Instead of global average pooling and a single linear layer
                                        there will be three linear layers of decreasing size.
            use_dropout         (bool): Use dropout after the residual layers.
        """
        super(ResNext50, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dropout is not in the original ResNext paper
        self.use_dropout = use_dropout

        self.in_dimensions = in_dimensions
        self.out_classes = out_classes

        # model a and model b should end up on different GPUs with data transfers between the two
        # pathways.
        self.model = nn.ModuleList()
        # The number of channels increases compared to the shortcut path. The original input can
        # either be zero-padded before adding to the output of the convolutions, projections can be
        # used to increase dimensions, or all shortcuts can be projections. The second option was
        # used in the original paper (see section "Identity vs. Projection Shortcuts").
        self.shortcut_projections = nn.ModuleList()

        self.initializeSizes()

        out_size = in_dimensions[1:]
        # Visualization setup requires the internal sizes of the feature outputs.
        self.output_sizes = [out_size]

        # Initialize in a no_grad section so that we can fill in some initial values for the bias
        # tensors.
        with torch.no_grad():
            initial_block = []
            initial_block.append(nn.Conv2d(
                in_channels=self.channels[0][0], out_channels=self.channels[1][0],
                kernel_size=self.kernels[0], stride=self.strides[0], padding=self.padding[0]))
            initial_block[-1].bias.fill_(0)
            initial_block[-1].weight.fill_(math.sqrt(self.channels[0][0] * self.kernels[0]**2))
            out_size = (int((out_size[0] + 2 * self.padding[0] - self.kernels[0])/self.strides[0] + 1),
                        int((out_size[1] + 2 * self.padding[0] - self.kernels[0])/self.strides[0] + 1))

            # Batch norm comes before the activation layer
            initial_block.append(nn.BatchNorm2d(self.channels[1][0]))
            initial_block.append(nn.ReLU())
            initial_block.append(nn.MaxPool2d(kernel_size=3, stride=2))
            out_size = (int((out_size[0] - 3)/2 + 1),
                        int((out_size[1] - 3)/2 + 1))
            # The initial 7x7 convolution and max pool are combined in the visualization phase.
            self.output_sizes.append(out_size)
            self.model.append(nn.Sequential(*initial_block))
            self.shortcut_projections.append(None)

            out_size = self.createInternalResLayers(out_size)

            # Now a global average pooling layer before the linear layer for classification.
            # Linear layers accept the flattened feature maps.
            # TODO Should the bias of the linear layer be set to something in specific?
            if not expanded_linear:
                self.classifier = torch.nn.Sequential(
                    nn.AvgPool2d(kernel_size=(out_size[0], out_size[1])),
                    nn.Flatten(),
                    nn.Linear(in_features=self.channels[-1][-1], out_features=out_classes)
                    # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                    # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
                )
            else:
                linear_input_size = out_size[0]*out_size[1]*self.channels[-1][-1]
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    self.createLinearLayer(num_inputs=linear_input_size, num_outputs=1024),
                    self.createLinearLayer(num_inputs=1024, num_outputs=512),
                    nn.Linear(in_features=512, out_features=self.out_classes)
                    # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                    # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
                )
                self.classifier[3].bias.fill_(1.)
            self.activation = nn.ReLU()

            self.createVisMaskLayers(self.output_sizes)

    #TODO AMP
    #@autocast()
    def forward(self, x):
        # The initial block of the model is not a residual layer, but the subsequent layers are
        # residual.
        x = self.model[0](x)
        for idx in range(1, len(self.model)):
            y = self.model[idx](x)
            # If the channels doubled then use a projection for the shortcut
            if self.shortcut_projections[idx] is not None:
                proj = self.shortcut_projections[idx](x)
            # Add the identity or projection to the output of the conv block and then send it
            # through an activation layer.
            x = self.activation(y + proj)

        x = self.classifier(x)
        return x

    def vis_forward(self, x):
        """Forward and calculate a visualization mask of the convolution layers."""
        conv_outputs = []
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        x = self.model[0](x)
        conv_outputs.append(x)
        for idx in range(1, len(self.model)):
            y = self.model[idx](x)
            # If the channels doubled then use a projection for the shortcut
            if self.shortcut_projections[idx] is not None:
                proj = self.shortcut_projections[idx](x)
            # Add the identity or projection to the output of the conv block and then send it
            # through an activation layer.
            x = self.activation(x + y)
            conv_outputs.append(x)

        # Go backwards to create the visualization mask
        mask = None
        for i, features in enumerate(reversed(conv_outputs)):
            avg_outputs = torch.mean(conv_outputs[-(1+i)], dim=1, keepdim=True)
            if mask is None:
                mask = self.vis_layers[-(1+i)](avg_outputs)
            else:
                mask = self.vis_layers[-(1+i)](mask * avg_outputs)
            # Keep the maximum value of the mask at 1
            mask = mask / mask.max()

        x = self.classifier(x)
        return x, mask


class ResNext18(ResNext50):
    """ResNext18. A little closer to the internal layers of the 50 than the original ResNet 18."""

    def initializeSizes(self):
        """
        Override this function to implement a different kind of residual network.
        """
        block1_channels = [64, 64, 64]
        block2_channels = [128, 128, 128]
        block3_channels = [256, 256, 256]
        block4_channels = [512, 512, 512]
        self.channels = ([self.in_dimensions[0]], [64], *([block1_channels] * 2), *([block2_channels] * 2),
                *([block3_channels] * 2), *([block4_channels] * 2))
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (7, *([3] * (2+2+2+2)))
        self.strides = (2, *([2, 1] * 4))
        self.padding = (3, *([1] * (2+2+2+2)))
        self.groups = (1, *([32] * (2+2+2+2)))
        # The first 7x7 kernel is followed by a 3x3 pooling layer, so the 7 needs to be
        # adjusted to a 9.
        self.vis_mask_sizes = (9, *([3] * (2+2+2+2)))
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))


class ResNext34(ResNext50):
    """ResNext34. A little closer to the internal layers of the 50 than the original ResNet 34."""

    def initializeSizes(self):
        """
        Override this function to implement a different kind of residual network.
        """
        block1_channels = [64, 64, 64]
        block2_channels = [128, 128, 128]
        block3_channels = [256, 256, 256]
        block4_channels = [512, 512, 512]
        self.channels = ([self.in_dimensions[0]], [64], *([block1_channels] * 3), *([block2_channels] * 4),
                *([block3_channels] * 6), *([block4_channels] * 3))
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (7, *([3] * (3+4+6+3)))
        self.strides = (2, *([2] + [1]*2), *([2] + [1]*3), *([2] + [1]*5), *([2] + [1]*2))
        self.padding = (3, *([1] * (3+4+6+3)))
        self.groups = (1, *([32] * (3+4+6+3)))
        # The first 7x7 kernel is followed by a 3x3 pooling layer, so the 7 needs to be
        # adjusted to a 9.
        self.vis_mask_sizes = (9, *([3] * (3+4+6+3)))
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))
