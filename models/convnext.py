"""
The ConvNeXt architecture is similar to previous convnet but was updated with many of the advances
found in recent papers (as of early 2022).
Networks are expanded by increasing the numbers of channel (denoted with C) or block (denoted with
B). Configurations suggested in the ConvNeXt paper are:
    ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)
    ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)
    ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
    ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)
    ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)
See: https://arxiv.org/pdf/2201.03545.pdf
"""

import itertools
import math
import torch
import torch.nn as nn

class PermuteModule(torch.nn.Module):
    """Making a permute module to keep the structure more clean."""
    def __init__(self, dims):
        super(PermuteModule, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class LayerScale(torch.nn.Module):
    """Some layers are scaled down. I would love to see a good justification for this. If feels like
    the same thing could be accomplished by warming up with a small learning rate, but it could be
    that numerical precision is not good enough to accomodate."""
    def __init__(self, channels, initial_scale=1e-6):
        super(LayerScale, self).__init__()
        self.scale = nn.Parameter(torch.ones((channels)) * initial_scale, requires_grad=True)

    def forward(self, x):
        return self.scale * x

class ChannelLayerNorm(torch.nn.Module):
    """Layer normalization is used heavily, but there is no channel first version in torch.

    Effectively, this is not too different from LocalResponseNormalization, which was used in
    AlexNet. Combined with the LayerScale operation seems to suggest that LocalResponseNormalization
    should be reinvestigated.
    """
    def __init__(self, channels, eps):
        super(ChannelLayerNorm, self).__init__()
        # Here is the ugliest possible implementation, involving the PermuteModule
        self.block = nn.Sequential(
            PermuteModule((0, 2, 3, 1)),
            nn.LayerNorm(normalized_shape=channels, eps=eps),
            PermuteModule((0, 3, 1, 2))
        )

    def forward(self, x):
        return self.block(x)

class ConvNextBase(nn.Module):
    """ConvNextBase."""

    def initializeWeights(self, module):
        """Default initialization applied to the convolution layers.

        The linear layers present are filling in for 1x1 convolutions in for tensors in a channels
        last view.
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # The paper uses trancated normal distribution with a standard deviation of 0.02.
            # The library used for the trancated normal sets the default truncations to -2 and 2,
            # which are 100 standard deviations away from the 0 mean. The function finishes by
            # clamping to the extreme values. I'll skip all of the math since the extreme cases
            # won't happen unless we continue running this program until long after our planet has
            # been consumed by the sun and just use the regular normal and then clamp it for fun.
            nn.init.normal_(module.weight, std=0.02)
            module.weight.clamp(min=-2, max=2)
            nn.init.constant_(module.bias, 0)

    def createResLayer(self, i, out_size=None):
        """
        Each convolution block consists of a 7x7 depthwise convolution, layer normalization, then a
        1x1 convolution to increase the number of channels by a factor of four, GELU activation
        function, and a 1x1 convolution to bottleneck the number of channels to the same as the
        input. Depthwise just means that the convolution is done across a channel, but not between
        channels.

        Actual downsampling occurs in different downsampling layers with a 2x2 conv with stride 2 on
        the image and a 1x1 conv with stride 2 on the shortcut. Layer normalization is used to
        stabilize training.

        Biases are initialized to 0 and weights are a truncated normal with std=0.02

        The official code for ConvNext says that blocks are done after permuting to (N,H,W,C)
        because it is a bit faster, but it is important to note that LayerNorm does not support
        a channel first arrangement. The closest thing might be local contrast norm, which was used
        back in AlexNet.
        
        Arguments:
            i          (int): Current layer index.
            out_size (tuple): Height and width of the current feature maps.
        Returns:
            tuple(nn.Sequential, tuple(int, int)): A tuple of the convolution layer and output size.
        """
        block = []


        # This is the depthwise convolution from the paper
        padding = self.padding[i]
        ksize = self.kernels[i]
        stride = self.strides[i]
        block.append(nn.Conv2d(
            in_channels=self.channels[i], out_channels=self.channels[i], kernel_size=ksize,
            padding=padding, groups=self.channels[i]))
        if out_size is not None:
            out_size = (int((out_size[0] + 2 * padding - ksize)/stride + 1),
                        int((out_size[1] + 2 * padding - ksize)/stride + 1))

        # View the tensor as channels last
        block.append(PermuteModule((0, 2, 3, 1)))
        block.append(nn.LayerNorm(normalized_shape=self.channels[i], eps=1e-6))
        # The 1x1 convolutions are equivalent to linear layers in the permuted space.
        # Recal that a 1x1 convolution's input is all of the channels at a particular pixel
        # location, e.g. it is fully connected across channels.
        block.append(nn.Linear(self.channels[i], 4*self.channels[i]))
        block.append(nn.GELU())
        block.append(nn.Linear(4*self.channels[i], self.channels[i]))

        # This scaling also assumes that channels are last since it is applying a learned scaling
        # per channel.
        if self.scale[i] is not None:
            block.append(LayerScale(self.channels[i], self.scale[i]))

        block.append(PermuteModule((0, 3, 1, 2)))

        return nn.Sequential(*block), out_size

    def createDownsample(self, i, out_size=None):
        """
        Create a downsampling layer that consists of a per-channel layer normalization then a
        downsampling convolution. Downsampling always used a 2x2 stride 2 convolution in the
        ConvNext paper.

        Arguments:
            i          (int): Current layer index.
            out_size (tuple): Height and width of the current feature maps.
        Returns:
            tuple(nn.Sequential, tuple(int, int)): A tuple of the convolution layer and output size.
        """
        padding = self.padding[i]
        ksize = self.kernels[i]
        stride = self.strides[i]
        layer = nn.Sequential(
            ChannelLayerNorm(channels=self.channels[i], eps=1e-6),
            nn.Conv2d(
                in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=ksize,
                padding=padding, stride=stride),
        )
        if out_size is not None:
            out_size = (int((out_size[0] + 2 * padding - ksize)/stride + 1),
                        int((out_size[1] + 2 * padding - ksize)/stride + 1))

        return layer, out_size

    def initializeSizes(self):
        """
        Override this function to implement different variants of the network.
        """
        layer_channels = [128, 256, 512, 1024]
        self.num_blocks = [3, 3, 27, 3]
        self.channels = [self.in_dimensions[0], 96]
        self.kernels = [4]
        self.strides = [4]
        self.padding = [0]
        self.scale = [None]
        self.downsampling = [True]
        for idx, (channels, blocks) in enumerate(zip(layer_channels, self.num_blocks)):
            self.channels += [channels] * blocks
            self.kernels += [7] * blocks
            self.strides += [1] * blocks
            self.padding += [3] * blocks
            self.scale += [1e-6] * blocks
            self.downsampling += [False] * blocks
            # Insert downsampling layers after the internal convolutions, although not after the
            # last one.
            if idx + 1 != len(layer_channels):
                self.channels.append(layer_channels[idx+1])
                self.kernels.append(2)
                self.strides.append(2)
                self.padding.append(0)
                self.scale.append(None)
                self.downsampling.append(True)

    def createInternalResLayers(self, out_size):
        """
        Override this function to implement different variants of the network.

        Arguments:
            out_size (list[int]): The width and height of the current feature maps.
        """
        # Create the convolutions and intermediate layers
        for i in range(1, len(self.kernels)):
            if self.downsampling[i]:
                # Conv2d and LayerNorm. Adjust the output size appropriately.
                layer, out_size = self.createDownsample(i, out_size)
            else:
                # Generally 7x7 Conv2d and two 1x1 convolutions
                layer, out_size = self.createResLayer(i, out_size)

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

    def __init__(self, in_dimensions, out_classes):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
        """
        super(ConvNextBase, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                in_channels=self.channels[0], out_channels=self.channels[1],
                kernel_size=self.kernels[0], stride=self.strides[0], padding=self.padding[0]))
            out_size = (int((out_size[0] + 2 * self.padding[0] - self.kernels[0])/self.strides[0] + 1),
                        int((out_size[1] + 2 * self.padding[0] - self.kernels[0])/self.strides[0] + 1))

            # TODO There should be a layer norm in the initial block. Wanna permute cludge?
            # View the tensor as channels last, norm, then permute back
            initial_block.append(ChannelLayerNorm(channels=self.channels[1], eps=1e-6))
            self.model.append(nn.Sequential(*initial_block))

            self.output_sizes.append(out_size)

            # Create the internal convolution layers
            out_size = self.createInternalResLayers(out_size)

            # Final normalization
            self.classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=(out_size[0], out_size[1])),
                nn.Flatten(),
                nn.LayerNorm(normalized_shape=self.channels[-1], eps=1e-6),
                nn.Linear(in_features=self.channels[-1], out_features=self.out_classes)
                # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
            )

            # Initialize weights
            self.model.apply(self.initializeWeights)
            self.classifier.apply(self.initializeWeights)

            # Ready the visualization mask
            # TODO
            #self.createVisMaskLayers(self.output_sizes)

    #TODO AMP
    #@autocast()
    def forward(self, x):
        for idx in range(len(self.model)):
            y = self.model[idx](x)
            # Add the skip if this is not a downsampling layer
            if not self.downsampling[idx]:
                y = x + y
            x = y

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

class ConvNextExtraTiny(ConvNextBase):
    """ConvNextExtraTiny. Wasn't in the paper, but my 7GBs of memory aren't enough :("""

    def initializeSizes(self):
        """
        Override this function to implement different variants of the network.
        """
        layer_channels = [96, 192, 384, 768]
        self.num_blocks = [3, 3, 6, 3]
        self.channels = [self.in_dimensions[0], 96]
        self.kernels = [4]
        self.strides = [4]
        self.padding = [0]
        self.scale = [None]
        self.downsampling = [True]
        for idx, (channels, blocks) in enumerate(zip(layer_channels, self.num_blocks)):
            self.channels += [channels] * blocks
            self.kernels += [7] * blocks
            self.strides += [1] * blocks
            self.padding += [3] * blocks
            self.scale += [1e-6] * blocks
            self.downsampling += [False] * blocks
            # Insert downsampling layers after the internal convolutions, although not after the
            # last one.
            if idx + 1 != len(layer_channels):
                self.channels.append(layer_channels[idx+1])
                self.kernels.append(2)
                self.strides.append(2)
                self.padding.append(0)
                self.scale.append(None)
                self.downsampling.append(True)


class ConvNextTiny(ConvNextBase):
    """ConvNextTiny. Smallest version of the model presented in the paper."""

    def initializeSizes(self):
        """
        Override this function to implement different variants of the network.
        """
        layer_channels = [96, 192, 384, 768]
        self.num_blocks = [3, 3, 9, 3]
        self.channels = [self.in_dimensions[0], 96]
        self.kernels = [4]
        self.strides = [4]
        self.padding = [0]
        self.scale = [None]
        self.downsampling = [True]
        for idx, (channels, blocks) in enumerate(zip(layer_channels, self.num_blocks)):
            self.channels += [channels] * blocks
            self.kernels += [7] * blocks
            self.strides += [1] * blocks
            self.padding += [3] * blocks
            self.scale += [1e-6] * blocks
            self.downsampling += [False] * blocks
            # Insert downsampling layers after the internal convolutions, although not after the
            # last one.
            if idx + 1 != len(layer_channels):
                self.channels.append(layer_channels[idx+1])
                self.kernels.append(2)
                self.strides.append(2)
                self.padding.append(0)
                self.scale.append(None)
                self.downsampling.append(True)

class ConvNextSmall(ConvNextBase):
    """ConvNextSmall. Second smallest version of the model presented in the paper."""

    def initializeSizes(self):
        """
        Override this function to implement different variants of the network.
        """
        layer_channels = [96, 192, 384, 768]
        self.num_blocks = [3, 3, 27, 3]
        self.channels = [self.in_dimensions[0], 96]
        self.kernels = [4]
        self.strides = [4]
        self.padding = [0]
        self.scale = [None]
        self.downsampling = [True]
        for idx, (channels, blocks) in enumerate(zip(layer_channels, self.num_blocks)):
            self.channels += [channels] * blocks
            self.kernels += [7] * blocks
            self.strides += [1] * blocks
            self.padding += [3] * blocks
            self.scale += [1e-6] * blocks
            self.downsampling += [False] * blocks
            # Insert downsampling layers after the internal convolutions, although not after the
            # last one.
            if idx + 1 != len(layer_channels):
                self.channels.append(layer_channels[idx+1])
                self.kernels.append(2)
                self.strides.append(2)
                self.padding.append(0)
                self.scale.append(None)
                self.downsampling.append(True)

class ConvNextLarge(ConvNextBase):
    """ConvNextLarge. Second largest version of the model presented in the paper."""

    def initializeSizes(self):
        """
        Override this function to implement different variants of the network.
        """
        layer_channels = [192, 384, 768, 1536]
        self.num_blocks = [3, 3, 27, 3]
        self.channels = [self.in_dimensions[0], 96]
        self.kernels = [4]
        self.strides = [4]
        self.padding = [0]
        self.scale = [None]
        self.downsampling = [True]
        for idx, (channels, blocks) in enumerate(zip(layer_channels, self.num_blocks)):
            self.channels += [channels] * blocks
            self.kernels += [7] * blocks
            self.strides += [1] * blocks
            self.padding += [3] * blocks
            self.scale += [1e-6] * blocks
            self.downsampling += [False] * blocks
            # Insert downsampling layers after the internal convolutions, although not after the
            # last one.
            if idx + 1 != len(layer_channels):
                self.channels.append(layer_channels[idx+1])
                self.kernels.append(2)
                self.strides.append(2)
                self.padding.append(0)
                self.scale.append(None)
                self.downsampling.append(True)

class ConvNextExtraLarge(ConvNextBase):
    """ConvNextExtraLarge. Largest version of the model presented in the paper."""

    def initializeSizes(self):
        """
        Override this function to implement different variants of the network.
        """
        layer_channels = [256, 512, 1024, 2048]
        self.num_blocks = [3, 3, 27, 3]
        self.channels = [self.in_dimensions[0], 96]
        self.kernels = [4]
        self.strides = [4]
        self.padding = [0]
        self.scale = [None]
        self.downsampling = [True]
        for idx, (channels, blocks) in enumerate(zip(layer_channels, self.num_blocks)):
            self.channels += [channels] * blocks
            self.kernels += [7] * blocks
            self.strides += [1] * blocks
            self.padding += [3] * blocks
            self.scale += [1e-6] * blocks
            self.downsampling += [False] * blocks
            # Insert downsampling layers after the internal convolutions, although not after the
            # last one.
            if idx + 1 != len(layer_channels):
                self.channels.append(layer_channels[idx+1])
                self.kernels.append(2)
                self.strides.append(2)
                self.padding.append(0)
                self.scale.append(None)
                self.downsampling.append(True)

