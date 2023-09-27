"""
The model that appears chronologically after VGG, the basic ResNet architecture has remained
popular. The major architectural additional is a "skip" layer where after every pair of convolutions
the input is added back into the convolution output. This improves gradient propagation through the
network and enables deeper networks with better results. The number of features maps generally
progresses from in order through 64, 128, 256, 512, 1024, and 2048. There are 18, 34, 50, 101, and
152 layer versions in the original paper. The 18 variant has 2 pairs each of 64, 128, 256, and 512
feature maps. The 50, 101, and 152 layer networks have three convolutions per skip layer instead of
two, but use 1x1 convolutions around the 3x3 convolutions used in the smaller networks.
See: https://arxiv.org/abs/1512.03385 for the architecture.
See: https://arxiv.org/abs/1502.01852 for the weight initialization.
"""

import itertools
import math
import torch
import torch.nn as nn


class ResNet18(nn.Module):
    """ResNet18. Generally, experiments show this is no better than a non-residual network."""

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
        for j in range(2):
            input_size = self.channels[i]
            out_channels = self.channels[i]
            # The activation layer goes between convolutions in the block
            if j > 0:
                block.append(nn.LeakyReLU())
                out_channels = self.channels[i+1]
                stride = self.strides[i]
            else:
                # The stride is always one inside of a block.
                stride = 1
            # The outputs of the previous layer are combined if this layer consumes data from the other
            # GPU as well.
            block.append(nn.Conv2d(
                in_channels=input_size, out_channels=out_channels,
                kernel_size=self.kernels[i], stride=stride, padding=self.padding[i]))
            block[-1].bias.fill_(0)
            block[-1].weight.fill_(math.sqrt(input_size * self.kernels[i]**2))
            if out_size is not None:
                out_size = (int((out_size[0] + 2 * self.padding[i] - self.kernels[i])/stride + 1),
                            int((out_size[1] + 2 * self.padding[i] - self.kernels[i])/stride + 1))
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
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )
        layer[0].bias.fill_(1.)
        return layer

    def initializeSizes(self):
        """
        Override this function to implement a different kind of residual network.
        """
        self.channels = (self.in_dimensions[0], 64, 64, 128, 128, 256, 256, 512, 512)
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (7, 3, 3, 3, 3, 3, 3, 3)
        self.strides = (2, 1, 2, 1, 2, 1, 2, 1)
        self.padding = (3, 1, 1, 1, 1, 1, 1, 1)
        # The sizes of the consecutive 3x3 convolutions in the residual layers are equivalent to a
        # 5x5 convolution.
        # The first 7x7 kernel is followed by a 3x3 pooling layer, so the 7 needs to be
        # adjusted to a 9.
        self.vis_mask_sizes = (9, 5, 5, 5, 5, 5, 5, 5)
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))

    def createInternalResLayers(self, out_size):
        """
        Override this function to implement a different kind of residual network.

        Arguments:
            out_size (list[int]): The width and height of the current feature maps.
        """
        # Now the residual layers
        for i in range(1, len(self.kernels)):
            # TODO FIXME Let's just 1x1 stride 2 with the proper number of output channels. It
            # is easier than messing with the size of the input layer while also messing with
            # this.
            if self.channels[i] < self.channels[i+1]:
                projection = nn.Conv2d(in_channels=self.channels[i],
                    out_channels=self.channels[i], stride=2, padding=0, kernel_size=1)
                downscale = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
                self.shortcut_projections.append(nn.ModuleList([projection, downscale]))
            else:
                self.shortcut_projections.append(None)
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

    def __init__(self, in_dimensions, out_classes, expanded_linear=False, vector_input_size=0):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
            expanded_linear     (bool): True to expand the linear layers from the initial paper.
                                        Instead of global average pooling and a single linear layer
                                        there will be three linear layers of decreasing size.
            vector_input_size    (int): The number of vector inputs to the linear layers.
        """
        super(ResNet18, self).__init__()
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
            initial_block[-1].bias.fill_(0)
            initial_block[-1].weight.fill_(math.sqrt(self.channels[0] * self.kernels[0]**2))
            out_size = (int((out_size[0] + 2 * self.padding[0] - self.kernels[0])/self.strides[0] + 1),
                        int((out_size[1] + 2 * self.padding[0] - self.kernels[0])/self.strides[0] + 1))

            # Batch norm comes before the activation layer
            initial_block.append(nn.BatchNorm2d(self.channels[1]))
            initial_block.append(nn.LeakyReLU())
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
                self.neck = torch.nn.Sequential(
                    nn.AvgPool2d(kernel_size=(out_size[0], out_size[1])),
                    nn.Flatten()
                )
                self.classifier = torch.nn.Sequential(
                    nn.Linear(in_features=self.channels[-1] + vector_input_size, out_features=out_classes)
                # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
                )
            else:
                self.neck = nn.Flatten()
                linear_input_size = out_size[0]*out_size[1]*self.channels[-1] + vector_input_size
                self.classifier = nn.Sequential(
                    self.createLinearLayer(num_inputs=linear_input_size, num_outputs=1024),
                    self.createLinearLayer(num_inputs=1024, num_outputs=512),
                    nn.Linear( in_features=512, out_features=self.out_classes)
                    # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                    # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
                )
                self.classifier[2].bias.fill_(1.)
            self.activation = nn.LeakyReLU()

            self.createVisMaskLayers(self.output_sizes)

    #TODO AMP
    #@autocast()
    def forward(self, x, vector_input=None):
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        x = self.model[0](x)
        for idx in range(1, len(self.model)):
            y = self.model[idx](x)
            # If the channels doubled then use a projection for the shortcut
            if self.shortcut_projections[idx] is not None:
                proj = self.shortcut_projections[idx][0](x)
                downscale = self.shortcut_projections[idx][1](x)
                x = torch.cat((downscale, proj), dim=1)
            # Add the shortcut and the output of the convolutions, then pass through an activation
            # layer
            x = self.activation(x + y)

        # Flatten, with other possible processing/pooling
        x = self.neck(x)

        if vector_input is not None:
            x = torch.cat((x, vector_input), dim=1)

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
                proj = self.shortcut_projections[idx][0](x)
                downscale = self.shortcut_projections[idx][1](x)
                x = torch.cat((downscale, proj), dim=1)
            # Add the shortcut and the output of the convolutions, then pass through an activation
            # layer
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

        # Flatten, with other possible processing/pooling
        x = self.neck(x)

        if vector_input is not None:
            x = torch.cat((x, vector_input), dim=1)

        x = self.classifier(x)
        return x, mask



class ResNet34(ResNet18):
    """ResNet34."""

    def initializeSizes(self):
        """
        Override this function to implement a different kind of residual network.
        """
        self.channels = (self.in_dimensions[0], 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512)
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
        self.strides = (2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1)
        self.padding = (3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        # The sizes of the consecutive 3x3 convolutions in the residual layers are equivalent to a
        # 5x5 convolution
        # The first 7x7 kernel is followed by a 3x3 pooling layer, so the 7 needs to be
        # adjusted to a 9.
        self.vis_mask_sizes = (9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))

    def createInternalResLayers(self, out_size):
        """
        Override this function to implement a different kind of residual network.

        Arguments:
            out_size (list[int]): The width and height of the current feature maps.
        """
        # Now the residual layers
        for i in range(1, len(self.kernels)):
            # TODO FIXME Let's just 1x1 stride 2 with the proper number of output channels. It
            # is easier than messing with the size of the input layer while also messing with
            # this.
            if self.channels[i] < self.channels[i+1]:
                projection = nn.Conv2d(in_channels=self.channels[i],
                    out_channels=self.channels[i], stride=2, padding=0, kernel_size=1)
                downscale = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
                self.shortcut_projections.append(nn.ModuleList([projection, downscale]))
            else:
                self.shortcut_projections.append(None)
            layer, out_size = self.createResLayer(i, out_size)
            self.output_sizes.append(out_size)
            self.model.append(layer)
        return out_size
