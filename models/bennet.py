"""
This is a model that will be similar to the PilotNet type architectures that I am more used to. This
is for my own ease of debugging as well as the model's effectiveness.
This model emphasises dropout and generalization and thus has more stability in performance on
evaluation sets, which may make it a better choice than some very large networks when the amount of
data available is limited. It may take more training iterations to reach a particular performance
level when compared with Alexnet or similar networks.
See: https://arxiv.org/pdf/2010.08776.pdf section 6 for a very bad, but colorful, drawing.
"""

import itertools
import math
import torch
import torch.nn as nn
import torchvision.ops as ops


class BenNet(nn.Module):
    """A small residual network."""

    def createResLayer(self, layer_idx, out_size=None):
        """
        This is similar to the createResLayer function in resnet.py.
        It creates the non-skip pathway of the residual layers, which consist of two convolutions.
        Arguments:
            layer_idx  (int): Current layer index.
            out_size (tuple): Height and width of the current feature maps.
        Returns:
            tuple(nn.Sequential, tuple(int, int)): A tuple of the convolution layer and output size.
        """
        block = []
        for residual_conv_index in range(2):
            input_size = self.channels[layer_idx]
            out_channels = self.channels[layer_idx]
            if residual_conv_index == 0:
                # The stride is always one to avoid shrinking the image before the layer convolution
                # of the residual
                stride = 1
                # If this is the very first convolution being done to the input image then expand
                # the channels in the first convolution.
                if layer_idx == 0:
                    out_channels = self.channels[layer_idx+1]
            else:
                out_channels = self.channels[layer_idx+1]
                stride = self.strides[layer_idx]
                # On the first convolution done to the image the channels were expanded in the first
                # convolution of the residual layer
                if layer_idx == 0:
                    input_size = self.channels[layer_idx+1]
            # The outputs of the previous layer are combined if this layer consumes data from the other
            # GPU as well.
            block.append(nn.Conv2d(
                in_channels=input_size, out_channels=out_channels,
                kernel_size=self.kernels[layer_idx], stride=stride, padding=self.padding[layer_idx]))
            #block[-1].bias.fill_(0)
            #block[-1].weight.fill_(math.sqrt(input_size * self.kernels[layer_idx]**2))
            #torch.nn.init.uniform_(block[-1].bias, 0.5, 1.0)
            #torch.nn.init.normal_(block[-1].weight, 0.0, 0.1)
            if out_size is not None:
                out_size = (int((out_size[0] + 2 * self.padding[layer_idx] - self.kernels[layer_idx])/stride + 1),
                            int((out_size[1] + 2 * self.padding[layer_idx] - self.kernels[layer_idx])/stride + 1))
            # Batch norm comes after the activation layer
            block.append(nn.ReLU())
            block.append(nn.BatchNorm2d(out_channels))
        if self.stochastic_depth[layer_idx] is not None:
            block.append(ops.StochasticDepth(p=self.stochastic_depth[layer_idx], mode='batch'))

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
        )
        #torch.nn.init.uniform_(layer[0].bias, 0.5, 1.0)
        #torch.nn.init.normal_(layer[0].weight, 0.0, 1.0)
        return layer

    def initializeSizes(self):
        """
        Override this function to change internal layer parameters.
        """
        self.channels = (self.in_dimensions[0], 48, 72, 96, 128, 192, 256, 256)
        #self.stochastic_depth = [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        self.stochastic_depth = [None, None, None, None, None, None, None]
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (3, 3, 3, 3, 3, 3, 3)
        self.strides = (2, 2, 2, 2, 2, 2, 2)
        self.padding = (1, 1, 1, 1, 1, 1, 1)
        self.non_res_layers = 2
        # The sizes of the consecutive 3x3 convolutions in the residual layers are equivalent to a
        # 5x5 convolution
        self.vis_mask_sizes = (5, 5, 5, 5, 5, 3, 3)
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))

    def createInternalResLayers(self, out_size):
        """
        Override this function to implement a different kind of residual network.

        Arguments:
            out_size (list[int]): The width and height of feature maps after convolutions.
        """
        # Create the residual layers
        for i in range(len(self.kernels) - self.non_res_layers):
            # The skip projection using a 1x1 kernel
            projection = nn.Conv2d(in_channels=self.channels[i],
                out_channels=self.channels[i+1], stride=2, padding=0, kernel_size=1)
            self.shortcut_projections.append(
                nn.Sequential(projection, nn.BatchNorm2d(self.channels[i+1])))
            self.post_residuals.append(
                nn.Sequential((nn.ReLU()), nn.Dropout2d(p=0.0)))

            # The non-skip layer with two convolutions.
            layer, out_size = self.createResLayer(i, out_size)
            self.output_sizes.append(out_size)
            self.model.append(layer)

        # Create the regular, non-residual convolution layers
        for i in range(len(self.kernels) - self.non_res_layers, len(self.kernels)):
            block = []

            input_size = self.channels[i]
            out_channels = self.channels[i + 1]
            block.append(nn.Conv2d(
                in_channels=input_size, out_channels=out_channels,
                kernel_size=self.kernels[i], stride=self.strides[i], padding=self.padding[i]))
            #block[-1].bias.fill_(0)
            #block[-1].weight.fill_(math.sqrt(input_size * self.kernels[i]**2))
            #torch.nn.init.uniform_(block[-1].bias, 0.5, 1.0)
            #torch.nn.init.normal_(block[-1].weight, 0.0, 0.1)
            out_size = (int((out_size[0] + 2 * self.padding[i] - self.kernels[i])/self.strides[i] + 1),
                        int((out_size[1] + 2 * self.padding[i] - self.kernels[i])/self.strides[i] + 1))
            self.output_sizes.append(out_size)
            # Batch norm comes after the activation layer
            block.append(nn.ReLU())
            block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.Dropout2d(p=0.0))
            self.model.append(nn.Sequential(*block))

        # The output size of the final channels
        return out_size

    def createVisMaskLayers(self, output_sizes):
        """
        Arguments:
            output_sizes (list[(int, int)]): Feature map dimensions.
        """
        self.vis_layers = nn.ModuleList()
        for i in range(len(self.vis_mask_sizes)):
            # The default output size of the ConvTranspose2d is the minimum dimension of the source
            # but the source could have had a larger dimension.
            # For example, if the dimension at layer i+1 is n with a stride of x then the source
            # could have had a dimension as small as n*x-(x-1) or as large as n*x.
            # To resolve this ambiguity we need to add in specific padding.
            max_source = (
                self.strides[i] * output_sizes[i+1][0], self.strides[i] * output_sizes[i+1][1])
            min_source = (
                max_source[0] - (self.strides[i] - 1), max_source[1] - (self.strides[i] - 1))
            output_padding = ( 
                output_sizes[i][0] - min_source[0], output_sizes[i][1] - min_source[1])
            padding = self.vis_mask_sizes[i] // 2
            self.vis_layers.append(nn.ConvTranspose2d(
                in_channels=1, out_channels=1, kernel_size=self.vis_mask_sizes[i],
                stride=self.strides[i], padding=padding, output_padding=output_padding, bias=False))
            self.vis_layers[-1].weight.requires_grad_(False)
            self.vis_layers[-1].weight.fill_(1.)

    def __init__(self, in_dimensions, out_classes, vector_input_size=0):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
            vector_input_size    (int): The number of vector inputs to the linear layers.
        """
        super(BenNet, self).__init__()
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
        self.post_residuals = nn.ModuleList()

        self.initializeSizes()

        out_size = in_dimensions[1:]
        # Visualization setup requires the internal sizes of the feature outputs.
        self.output_sizes = [out_size]

        # Initialize in a no_grad section so that we can fill in some initial values for the bias
        # tensors.
        with torch.no_grad():
            self.initial_batch_norm = nn.Sequential(nn.BatchNorm2d(in_dimensions[0]))
            out_size = self.createInternalResLayers(out_size)

            self.neck = nn.Sequential(nn.Flatten())

            # Linear layers accept the flattened feature maps.
            linear_input_size = out_size[0]*out_size[1]*self.channels[-1] + vector_input_size
            self.classifier = nn.Sequential(
                self.createLinearLayer(num_inputs=linear_input_size, num_outputs=256),
                #nn.Dropout1d(p=0.25),
                self.createLinearLayer(num_inputs=256, num_outputs=256),
                self.createLinearLayer(num_inputs=256, num_outputs=128),
                self.createLinearLayer(num_inputs=128, num_outputs=96),
                nn.Linear(in_features=96, out_features=self.out_classes)
                # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
            )

            self.createVisMaskLayers(self.output_sizes)

    #TODO AMP
    #@autocast()
    def forward(self, x, vector_input=None):
        x = self.initial_batch_norm(x)
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        for idx in range(len(self.model) - self.non_res_layers):
            y = self.model[idx](x)
            proj = self.shortcut_projections[idx](x)
            # Add the shortcut and the output of the convolutions, then pass through activation and
            # dropout layers.
            x = self.post_residuals[idx](y + proj)

        for layer in self.model[-self.non_res_layers:]:
            x = layer(x)

        # Flatten
        x = self.neck(x)

        if vector_input is not None:
            x = torch.cat((x, vector_input), dim=1)

        x = self.classifier(x)
        return x

    def vis_forward(self, x, vector_input=None):
        """Forward and calculate a visualization mask of the convolution layers."""
        x = self.initial_batch_norm(x)
        conv_outputs = []
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        for idx in range(len(self.model) - self.non_res_layers):
            y = self.model[idx](x)
            proj = self.shortcut_projections[idx](x)
            # Add the shortcut and the output of the convolutions, then pass through activation and
            # dropout layers.
            x = self.post_residuals[idx](y + proj)
            conv_outputs.append(x)

        for layer in self.model[-self.non_res_layers:]:
            x = layer(x)
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

        # Flatten
        x = self.neck(x)

        if vector_input is not None:
            x = torch.cat((x, vector_input), dim=1)

        x = self.classifier(x)
        return x, mask

    def produceFeatureMaps(self, x):
        """Produce and return max pooled feature maps from all of the convolution layers."""
        maps = []
        x = self.initial_batch_norm(x)
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        for idx in range(len(self.model) - self.non_res_layers):
            y = self.model[idx](x)
            proj = self.shortcut_projections[idx](x)
            # Add the shortcut and the output of the convolutions, then pass through activation and
            # dropout layers.
            x = self.post_residuals[idx](y + proj)
            maps.append(x)

        for layer in self.model[-self.non_res_layers:]:
            x = layer(x)
            maps.append(x)

        return maps


class CompactingBenNet(BenNet):
    """A version of the network that squeezes features vertically and horizontally before
    flattening."""

    def __init__(self, in_dimensions, out_classes, vector_input_size=0):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
            vector_input_size    (int): The number of vector inputs to the linear layers.
        """
        super(CompactingBenNet, self).__init__(in_dimensions=in_dimensions, out_classes=out_classes,
            vector_input_size=vector_input_size)

        # TODO Replace the neck with a parallel horizontal and vertical convolutions that remove
        # some of the spatial information from the remaining pixels in the feature maps
        # This is to make it easier to do operations upon the embedded space.
        # Then replace the first linear layer of the network with a layer that has the correct input
        # size
        self.height_collapsing_conv = nn.Conv2d(
            in_channels=self.channels[-1], out_channels=self.channels[-1],
            kernel_size=(self.output_sizes[-1][0], 1))
        self.width_collapsing_conv = nn.Conv2d(
            in_channels=self.channels[-1], out_channels=self.channels[-1],
            kernel_size=(1, self.output_sizes[-1][1]))

        # The new input has a single entry per row and column for each channel
        linear_input_size = (self.output_sizes[-1][0] + self.output_sizes[-1][1]) * self.channels[-1] + vector_input_size
        self.classifier[0] = self.createLinearLayer(num_inputs=linear_input_size, num_outputs=256)

    #TODO AMP
    #@autocast()
    def forward(self, x, vector_input=None):
        x = self.initial_batch_norm(x)
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        for idx in range(len(self.model) - self.non_res_layers):
            y = self.model[idx](x)
            proj = self.shortcut_projections[idx](x)
            # Add the shortcut and the output of the convolutions, then pass through activation and
            # dropout layers.
            x = self.post_residuals[idx](y + proj)

        for layer in self.model[-self.non_res_layers:]:
            x = layer(x)

        vertical_features = self.height_collapsing_conv(x)
        horizontal_features = self.width_collapsing_conv(x)

        # Flatten
        x = torch.cat((self.neck(vertical_features), self.neck(horizontal_features)), dim=1)

        if vector_input is not None:
            x = torch.cat((x, vector_input), dim=1)

        x = self.classifier(x)
        return x

    def vis_forward(self, x, vector_input=None):
        """Forward and calculate a visualization mask of the convolution layers."""
        x = self.initial_batch_norm(x)
        conv_outputs = []
        # The initial block of the model is not a residual layer, but then there is a skip
        # connection for every pair of layers after that.
        for idx in range(len(self.model) - self.non_res_layers):
            y = self.model[idx](x)
            proj = self.shortcut_projections[idx](x)
            # Add the shortcut and the output of the convolutions, then pass through activation and
            # dropout layers.
            x = self.post_residuals[idx](y + proj)
            conv_outputs.append(x)

        for layer in self.model[-self.non_res_layers:]:
            x = layer(x)
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

        vertical_features = self.height_collapsing_conv(x)
        horizontal_features = self.width_collapsing_conv(x)

        # Flatten
        x = self.neck(torch.cat(vertical_features, horizontal_features), dim=1)

        if vector_input is not None:
            x = torch.cat((x, vector_input), dim=1)

        x = self.classifier(x)
        return x, mask


class TinyBenNet(BenNet):
    """A version of the network for tiny images, like MNIST."""


    def initializeSizes(self):
        """
        Override this function to change internal layer parameters.
        """
        self.channels = (self.in_dimensions[0], 48, 72, 96, 128)
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (3, 3, 3, 3)
        self.strides = (2, 2, 1, 1)
        self.padding = (1, 1, 1, 1)
        self.non_res_layers = 2
        # The sizes of the consecutive 3x3 convolutions in the residual layers are equivalent to a
        # 5x5 convolution
        self.vis_mask_sizes = (5, 5, 3, 3)
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))
