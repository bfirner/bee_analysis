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

from models.modules import MaxThresholding, PresolvedConv2d, PresolvedLinear


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
                out_channels=self.channels[i+1], stride=self.strides[i], padding=0, kernel_size=1)
            self.shortcut_projections.append(
                nn.Sequential(projection, nn.BatchNorm2d(self.channels[i+1])))
            self.post_residuals.append(
                nn.Sequential((nn.ReLU()), nn.Dropout2d(p=0.5)))

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
            # TODO FIXME Trying no last batch norm to allow more discretized signal to escape the
            # convolutions. Early experiments show a small shift towards the local mean, e.g.
            # reduced fitting to the training set.
            if i+2 < len(self.kernels):
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.Dropout2d(p=0.5))
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
            #self.initial_batch_norm = nn.Sequential(nn.BatchNorm2d(in_dimensions[0]))
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

            self.vector_preprocess = nn.Sequential()
            # Almost always drop out the vector inputs so that they only have a very slight
            # modulatory effect upon training but can still be used during inference.
            #self.vector_preprocess = nn.Sequential(nn.Dropout1d(p=0.90))

            self.createVisMaskLayers(self.output_sizes)

    def normalizeVectorInputs(self, vector_means, vector_stddevs):
        """Normalize vector inputs by setting weights to 1/stddevs and and biases to -means"""

        # The vector inputs are appended to the flattened outputs of the convolutional layer.
        # Implement this in the first layer after vector input concatenation
        # TODO This may not work nicely with the vector_preprocess layer, if that is doing anything
        with torch.no_grad():
            # Bias to +1 because of ReLUs
            self.classifier[0][0].bias[-len(vector_means):] = 1.0 + -1 * torch.tensor(vector_means).to(self.classifier[0][0].bias.device)
            self.classifier[0][0].weight[:,-len(vector_stddevs):] = 1.0 / torch.tensor(vector_stddevs).to(self.classifier[0][0].weight.device)

    #TODO AMP
    #@autocast()
    def forward(self, x, vector_input=None):
        #x = self.initial_batch_norm(x)
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
            x = torch.cat((x, self.vector_preprocess(vector_input)), dim=1)

        x = self.classifier(x)
        return x

    def vis_forward(self, x, vector_input=None):
        """Forward and calculate a visualization mask of the convolution layers."""
        #x = self.initial_batch_norm(x)
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
            x = torch.cat((x, self.vector_preprocess(vector_input)), dim=1)

        x = self.classifier(x)
        return x, mask

    def produceFeatureMaps(self, x):
        """Produce and return max pooled feature maps from all of the convolution layers."""
        maps = []
        #x = self.initial_batch_norm(x)
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

    def forwardToFeatures(self, x):
        """Produce and return the feature maps before the linear layers."""
        #x = self.initial_batch_norm(x)
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

        return x


class CompactingBenNet(BenNet):
    """A version of the network that squeezes features vertically and horizontally before
    flattening."""

    def initializeSizes(self):
        """
        Preserve the spatial dimensions and sacrifice channel counts.
        """
        self.channels = (self.in_dimensions[0], 24, 36, 48, 64, 128, 128, 128)
        #self.stochastic_depth = [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        self.stochastic_depth = [None, None, None, None, None, None, None]
        # The first layer quickly reduces the size of feature maps.
        # The stride of 2 is used whenever the feature map size doubles to keep computation roughly
        # the same.
        self.kernels = (3, 3, 3, 3, 3, 3, 3)
        self.strides = (2, 2, 1, 1, 1, 1, 1)
        self.padding = (1, 1, 1, 1, 1, 1, 1)
        self.non_res_layers = 2
        # The sizes of the consecutive 3x3 convolutions in the residual layers are equivalent to a
        # 5x5 convolution
        self.vis_mask_sizes = (5, 5, 3, 3, 3, 3, 3)
        assert(len(self.kernels) == len(self.strides) == len(self.padding) == len(self.vis_mask_sizes))

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
        #self.height_collapsing_conv = nn.Conv2d(
        #    in_channels=self.channels[-1], out_channels=self.channels[-1],
        #    kernel_size=(self.output_sizes[-1][0], 1))
        #self.width_collapsing_conv = nn.Conv2d(
        #    in_channels=self.channels[-1], out_channels=self.channels[-1],
        #    kernel_size=(1, self.output_sizes[-1][1]))
        # TODO FIXME Trying to bottleneck the number of features, dividing by 2 and then 4 in the
        # compacting convolutions
        self.presolve = False
        if not self.presolve:
            self.neck = torch.nn.Sequential(
                # Collapse the height so that there is a single row per channel
                nn.Conv2d(
                    in_channels=self.channels[-1], out_channels=self.channels[-1]//2,
                    kernel_size=(self.output_sizes[-1][0], 1)),
                # Collapse the width so that there is a single pixel per channel
                nn.Conv2d(
                    in_channels=self.channels[-1]//2, out_channels=self.channels[-1]//4,
                    kernel_size=(1, self.output_sizes[-1][1])),
                nn.Flatten())

            # The new input has a single entry per row and column for each channel
            #linear_input_size = (self.output_sizes[-1][0] + self.output_sizes[-1][1]) * self.channels[-1] + vector_input_size
            linear_input_size = self.channels[-1]//4 + vector_input_size
            self.classifier[0] = self.createLinearLayer(num_inputs=linear_input_size, num_outputs=256)
        else:
            # Replace the last ReLU of the convolution block with a Sigmoid
            #self.model[-1][1] = torch.nn.Sigmoid()
            # We assume that the y solution and x solution will both appear in the features of
            # channel 0. The desire is that the feature map be 0 except where the tip of the robot
            # is present.
            # The bobot is approaching from the right. x in the image is -x of the robot.
            # y in the image is y of the robot.
            # Weights should end up with size: [out_channels, in_channels, height, width]
            original_height = in_dimensions[1]
            #y_scaling = self.output_sizes[-1][0] / original_height
            y_scaling = 1. / self.output_sizes[-1][0]

            original_width = in_dimensions[2]
            #x_scaling = self.output_sizes[-1][1] / original_width
            x_scaling = 1. / self.output_sizes[-1][1]

            vertical_weights = torch.cat((
                # This is the kernel across the y dimension to deduce the y offset
                torch.cat((
                    torch.arange(0., self.output_sizes[-1][0]).view(1, 1, -1, 1)*y_scaling*y_scaling,
                    #torch.zeros(self.output_sizes[-1][0]).view(1, 1, -1, 1)*y_scaling,
                    torch.zeros(1, self.channels[-1]-1, self.output_sizes[-1][0], 1),
                    ), dim=1
                ),
                # This is the kernel to pass through the values that will be used to deduce the x
                # offset
                torch.cat((
                    torch.ones(1, 1, self.output_sizes[-1][0], 1)*x_scaling*x_scaling,
                    #torch.zeros(1, 1, self.output_sizes[-1][0], 1),
                    torch.zeros(1, self.channels[-1]-1, self.output_sizes[-1][0], 1),
                    ), dim=1),
                ), dim=0)

            horizontal_weights = torch.cat((
                # This is the kernel to preserve the discovered y value in the first channel
                torch.cat((
                    torch.ones(1, 1, 1, self.output_sizes[-1][1]),
                    #torch.zeros(1, 1, 1, self.output_sizes[-1][1]),
                    torch.zeros(1, self.channels[-1]//2 - 1, 1, self.output_sizes[-1][1]),
                    ), dim=1
                ),
                # This is the kernel to deduce the x value from the second input channel
                # The x-scaling was already applies in the weights for the previous layer
                torch.cat((
                    torch.zeros(1, 1, 1, self.output_sizes[-1][1]),
                    # There should have only been one kernel that picked up values in the previous
                    # layer, but initially that won't be true. To compensate, scale by the number of
                    # possible nonzero outputs (which is the width)
                    torch.arange(self.output_sizes[-1][1], 0., -1.).view(1, 1, 1, -1)/self.output_sizes[-1][1],
                    #torch.ones(self.output_sizes[-1][1]).view(1, 1, 1, -1),
                    torch.zeros(1, self.channels[-1]//2 - 2, 1, self.output_sizes[-1][1]),
                    ), dim=1),
                ), dim=0)

            # The y values are 0 in the middle of the range of motion
            #vertical_bias = torch.tensor([-original_height/2, 0])
            vertical_bias = torch.tensor([0., 0.])
            horizontal_bias = torch.tensor([(-original_height/2.) * y_scaling * y_scaling,0.])
            self.neck = torch.nn.Sequential(
                # TODO This can probably be removed now that the bias was adjusted. Try adjusting
                # the weight again as well. The proper weight should account for about n^2 because
                # the sum of n elements is about n*(n+1)/2
                #MaxThresholding(alpha=0.25),
                # Collapse the height so that there is a single row per channel
                # This should also solve for the current y value
                PresolvedConv2d(
                    in_channels=self.channels[-1], out_channels=self.channels[-1]//2,
                    presolved_weights=vertical_weights, presolved_bias=vertical_bias,
                    kernel_size=(self.output_sizes[-1][0], 1)),
                # Collapse the width so that there is a single pixel per channel
                # This should also solve for the current x value and pass the solved y value through
                PresolvedConv2d(
                    in_channels=self.channels[-1]//2, out_channels=self.channels[-1]//4,
                    presolved_weights=horizontal_weights, presolved_bias=horizontal_bias,
                    kernel_size=(1, self.output_sizes[-1][1])),
                nn.Flatten())

            # The new input has a single entry per row and column for each channel
            linear_input_size = self.channels[-1]//4 + vector_input_size
            # There are multiple linear layers. The first one will just preserve the current value
            # solved for y and x
            linear_y = [1.] + [0.] * (linear_input_size-1)
            linear_x = [0.] + [1.] + [0.] * (linear_input_size-2)
            classifier_weights = torch.tensor([linear_y, linear_x])
            classifier_bias = torch.zeros(2)
            self.classifier[0] = nn.Sequential(
                PresolvedLinear(
                    in_features=linear_input_size, out_features=256,
                    presolved_weights=classifier_weights, presolved_bias=classifier_bias),
                nn.ReLU(),
            )

            linear_y = [1.] + [0.] * (256-1)
            linear_x = [0.] + [1.] + [0.] * (256-2)
            classifier_weights = torch.tensor([linear_y, linear_x])
            classifier_bias = torch.zeros(2)
            self.classifier[1] = nn.Sequential(
                PresolvedLinear(
                    in_features=256, out_features=256,
                    presolved_weights=classifier_weights, presolved_bias=classifier_bias),
                nn.ReLU(),
            )

            linear_y = [1.] + [0.] * (256-1)
            linear_x = [0.] + [1.] + [0.] * (256-2)
            classifier_weights = torch.tensor([linear_y, linear_x])
            classifier_bias = torch.zeros(2)
            self.classifier[2] = nn.Sequential(
                PresolvedLinear(
                    in_features=256, out_features=128,
                    presolved_weights=classifier_weights, presolved_bias=classifier_bias),
                nn.ReLU(),
            )

            linear_y = [1.] + [0.] * (128-1)
            linear_x = [0.] + [1.] + [0.] * (128-2)
            classifier_weights = torch.tensor([linear_y, linear_x])
            classifier_bias = torch.zeros(2)
            self.classifier[3] = nn.Sequential(
                PresolvedLinear(
                    in_features=128, out_features=96,
                    presolved_weights=classifier_weights, presolved_bias=classifier_bias),
                nn.ReLU(),
            )

            # Initialize the weights for layers after the presolved inputs to be quite low
            #torch.nn.init.uniform_(block[-1].bias, 0.5, 1.0)
            torch.nn.init.normal_(self.neck[0].learned_conv.weight, 0.0, 0.001)
            torch.nn.init.uniform_(self.neck[0].learned_conv.bias, -0.1, 0.1)
            torch.nn.init.normal_(self.neck[1].learned_conv.weight[:,0:2], 0.0, 0.00001)
            torch.nn.init.normal_(self.neck[1].learned_conv.weight[:,3:], 0.0, 0.01)
            #torch.nn.init.uniform_(self.neck[1].learned_conv.bias, -0.1, 0.1)
            torch.nn.init.normal_(self.classifier[0][0].learned_linear.weight[:,0:2], 0.0, 0.00001)
            #torch.nn.init.uniform_(self.classifier[0][0].learned_linear.bias, -0.1, 0.1)
            torch.nn.init.normal_(self.classifier[1][0].learned_linear.weight[:,0:2], 0.0, 0.00001)
            #torch.nn.init.uniform_(self.classifier[1][0].learned_linear.bias, -0.1, 0.1)
            torch.nn.init.normal_(self.classifier[2][0].learned_linear.weight[:,0:2], 0.0, 0.00001)
            #torch.nn.init.uniform_(self.classifier[2][0].learned_linear.bias, -0.1, 0.1)

    #TODO AMP
    #@autocast()
    def forward(self, x, vector_input=None):
        x = self.forwardToFeatures(x)

        if vector_input is not None:
            x = torch.cat((x, self.vector_preprocess(vector_input)), dim=1)

        x = self.classifier(x)
        return x

    def forwardToFeatures(self, x):
        """Produce and return the feature maps before the linear layers."""
        #x = self.#initial_batch_norm(x)
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

        #vertical_features = self.height_collapsing_conv(x)
        #horizontal_features = self.width_collapsing_conv(x)

        # Flatten
        #x = torch.cat((self.neck(vertical_features), self.neck(horizontal_features)), dim=1)
        # The features are flattened by the large horizontal and vertical kernels in the neck
        x = self.neck(x)
        #for i, layer in enumerate(self.neck):
        #    import sys
        #    x = layer(x)
        #    sys.stderr.write("max in layer {} is {}\n".format(i, torch.max(x.flatten(1), dim=1)[0]))
        #    print("max in layer {} is {}\n".format(i, torch.max(x.flatten(1), dim=1)[0]))

        return x

    def vis_forward(self, x, vector_input=None):
        """Forward and calculate a visualization mask of the convolution layers."""
        #x = self.initial_batch_norm(x)
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

        #vertical_features = self.height_collapsing_conv(x)
        #horizontal_features = self.width_collapsing_conv(x)

        # Flatten
        #x = self.neck(torch.cat(vertical_features, horizontal_features), dim=1)
        x = self.neck(x)

        if vector_input is not None:
            x = torch.cat((x, self.vector_preprocess(vector_input)), dim=1)

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
