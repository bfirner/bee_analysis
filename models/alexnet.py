"""
Network from "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky,
Sutskever, and Hinton. The input size is not the same so things will be a bit different, but it is
the same in spirit. There will be a series of convolution layers with pooling, ReLUs, and
normalizations in between and then a few linear layers to produce classification results.
See: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""

import itertools
import torch
import torch.nn as nn


class AlexLikeNet(nn.Module):
    """Five convolution layers and three linear layers."""


    def createConvLayer(self, i, out_size=None):
        """
        Arguments:
            i          (int): Current layer index.
            out_size (tuple): Height and width of the current feature maps.
        Returns:
            tuple(nn.Sequential, tuple(int, int)): A tuple of the convolution layer and output size.
        """
        block = []
        input_size = self.channels[i]
        # The outputs of the previous layer are combined if this layer consumes data from the other
        # GPU as well.
        if self.crossover[i]:
            input_size *= 2
        block.append(nn.Conv2d(
            in_channels=input_size, out_channels=self.channels[i+1],
            kernel_size=self.kernels[i], stride=self.strides[i], padding=self.padding[i]))
        block[-1].bias.fill_(self.biases[i])
        if out_size is not None:
            out_size = (int((out_size[0] + 2 * self.padding[i] - self.kernels[i])/self.strides[i] + 1),
                        int((out_size[1] + 2 * self.padding[i] - self.kernels[i])/self.strides[i] + 1))
        # From section 3.4, justified by a small classification gain.
        if self.pooling[i]:
            block.append(nn.MaxPool2d(kernel_size=3, stride=2))
            if out_size is not None:
                out_size = (int((out_size[0] - 3)/2 + 1),
                            int((out_size[1] - 3)/2 + 1))
        block.append(nn.ReLU())
        # See section 3.3
        # The parameters for local response normalization were determined through the evil
        # hyperparameter optimization, so their meaningfulness is open to question.
        if self.lrnorm[i]:
            block.append(nn.LocalResponseNorm(k=2, size=5, alpha=10e-4, beta=0.75))
        # Dropout was not used in the convolution layers of Alexnet. It was probably too new
        # at the time.
        # Neither was BatchNorm
        block.append(nn.BatchNorm2d(self.channels[i+1]))
        return nn.Sequential(*block), out_size

    def createLinearLayer(self, num_inputs, num_outputs):
        """
        Arguments:
            num_inputs  (int): Number of inputs to the linear layer.
            num_outputs (int): Number of outputs to the linear layer.
        Returns:
            nn.Sequential: The linear layer.
        """
        # The original Alexnet splits the linear layers over two GPUs so that the first two
        # linear layers are actually each a pair of linear layers with 2048 outputs. With 4096
        # inputs and 2048 outputs the layer's memory consumption is cut in half.
        layer = nn.Sequential(
            nn.Linear(
                in_features=num_inputs, out_features=num_outputs),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        layer[0].bias.fill_(1.)
        return layer

    def __init__(self, in_dimensions, out_classes, linear_size=2048):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
            linear_size          (int): The size of the linear layers. There are two at each depth.
        """
        super(AlexLikeNet, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_dimensions = in_dimensions
        self.out_classes = out_classes
        self.linear_size = linear_size

        # model a and model b should end up on different GPUs with data transfers between the two
        # pathways.
        self.model_a = nn.ModuleList()
        self.model_b = nn.ModuleList()

        # See section 3
        self.channels = (self.in_dimensions[0], 48, 128, 192, 192, 128)
        # The first layer quickly reduces the size of feature maps
        self.kernels = (11, 5, 3, 3, 3)
        self.strides = (4, 1, 1, 1, 1)
        self.padding = (3, 2, 1, 1, 1)
        self.pooling = (True, True, False, False, True)
        self.lrnorm = (True, True, False, False, False)
        # True when layer data is shared across the GPUs
        self.crossover = (False, False, True, False, False)
        # Some of the bias values are set to 1 to "accelerate the early stages of learning by
        # providing ReLUs with positive inputs". The others are 0. See section 5.
        self.biases = (0., 1., 0., 1., 1.)

        out_size = in_dimensions[1:]
        # Visualization setup requires the internal sizes of the feature outputs.
        self.output_sizes = [out_size]

        # Initialize in a no_grad section so that we can fill in some initial values for the bias
        # tensors.
        with torch.no_grad():
            for i in range(len(self.kernels)):
                layer, out_size = self.createConvLayer(i, out_size)
                self.output_sizes.append(out_size)
                self.model_a.append(layer)
                # The parallel pathway
                layer, _ = self.createConvLayer(i)
                self.model_b.append(layer)

            # 3 Linear layers accept the flattened feature maps.
            self.flatten = nn.Flatten()
            # The original Alexnet splits the linear layers over two GPUs so that the first two
            # linear layers are actually each a pair of linear layers with 2048 outputs. With 4096
            # inputs and 2048 outputs the layer's memory consumption is cut in half.
            linear_input_size = out_size[0]*out_size[1]*self.channels[-1]*2
            self.model_a.append(self.createLinearLayer(
                num_inputs=linear_input_size, num_outputs=self.linear_size))
            self.model_b.append(self.createLinearLayer(
                num_inputs=linear_input_size, num_outputs=self.linear_size))

            self.model_a.append(self.createLinearLayer(
                num_inputs=2*self.linear_size, num_outputs=self.linear_size))
            self.model_b.append(self.createLinearLayer(
                num_inputs=2*self.linear_size, num_outputs=self.linear_size))

            self.classifier = nn.Sequential(
                nn.Linear(
                    in_features=2*self.linear_size, out_features=self.out_classes),
                nn.ReLU(),
                # No softmax at the end. To train a single label classifier use CrossEntropyLoss
                # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
            )
            self.classifier[0].bias.fill_(1.)

            self.createVisMaskLayers(self.output_sizes)

    def createVisMaskLayers(self, output_sizes):
        """
        Arguments:
            output_sizes (list[(int, int)]): Feature map dimensions.
        """
        self.vis_layers = nn.ModuleList()
        for i in range(len(self.kernels)):
            stride = self.strides[i]
            ksize = self.kernels[i]
            padding = self.padding[i]
            if self.pooling[i]:
                # Pooling has a kernel size of 3 with a stride of 2.
                # TODO Make this a second upconv for simplicity? Nah, merge with this upconv
                # Merge with this kernel. The effective stride is doubled
                stride = stride * 2
                # The kernel's perceptive field goes up by one pixel on each size
                ksize += 2
                # Pooling was done without padding
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
            self.vis_layers.append(nn.ConvTranspose2d(
                in_channels=1, out_channels=1, kernel_size=ksize,
                stride=stride, padding=padding, output_padding=output_padding, bias=False))
            self.vis_layers[-1].weight.requires_grad_(False)
            self.vis_layers[-1].weight.fill_(1.)

    def forward(self, x):
        # TODO The two side of the network need to be transferred to different devices, and the
        # tensors need to be transferred back and forth as well.
        # Both pathways start with the same input
        y = x
        for i, (a, b) in enumerate(itertools.zip_longest(self.model_a, self.model_b)):
            x = a(x)
            y = b(y)
            # The outputs of the second layer are joined together to make the inputs of the third
            # layer. Remember that the first layer's index is 0.
            if (1 == i):
                x = torch.cat((x, y), dim=1)
                y = x
            # After the last convolution the output is flattened. For all linear layers the outputs
            # of the previous layer are concatenated.
            elif (4 == i):
                x = torch.cat((self.flatten(x), self.flatten(y)), dim=1)
                y = x
            elif (5 == i):
                x = torch.cat((x, y), dim=1)
                y = x
            elif (6 == i):
                x = torch.cat((x, y), dim=1)

        x = self.classifier(x)
        return x

    def vis_forward(self, x):
        """Forward and calculate a visualization mask of the convolution layers."""
        conv_outputs_a = []
        conv_outputs_b = []
        # The inputs will lag one index behind
        conv_inputs_a = []
        conv_inputs_b = []
        # Forward as usual, but store the convolution outputs for later backtracking to build the
        # visualization masks.

        y = x
        for i, (a, b) in enumerate(itertools.zip_longest(self.model_a, self.model_b)):
            x = a(x)
            y = b(y)
            if i < len(self.kernels):
                conv_outputs_a.append(x)
                conv_outputs_b.append(y)
            # The outputs of the second layer are joined together to make the inputs of the third
            # layer. Remember that the first layer's index is 0.
            if (1 == i):
                x = torch.cat((x, y), dim=1)
                y = x
            # After the last convolution the output is flattened. For all linear layers the outputs
            # of the previous layer are concatenated.
            elif (4 == i):
                x = torch.cat((self.flatten(x), self.flatten(y)), dim=1)
                y = x
            elif (5 == i):
                x = torch.cat((x, y), dim=1)
                y = x
            elif (6 == i):
                x = torch.cat((x, y), dim=1)
            # These are the actual inputs for the next layer
            #conv_inputs_a.append(x)
            #conv_inputs_b.append(y)

        # Go backwards to create the visualization mask
        mask_a = None
        mask_b = None
        # The following line is extremely pythonic.
        for i, (features_a, features_b) in enumerate(reversed(list(zip(conv_outputs_a, conv_outputs_b)))):
            # Flatten the current output
            avg_outputs_a = torch.mean(features_a, dim=1, keepdim=True)
            avg_outputs_b = torch.mean(features_b, dim=1, keepdim=True)
            # Multiply with the previous upconv features (if they exist) and then upconv
            if mask_a is None:
                mask_a = self.vis_layers[-(1+i)](avg_outputs_a)
                mask_b = self.vis_layers[-(1+i)](avg_outputs_b)
            else:
                mask_a = self.vis_layers[-(1+i)](mask_a * avg_outputs_a)
                mask_b = self.vis_layers[-(1+i)](mask_b * avg_outputs_b)
            # If the data crossed over from one route to another then masks from all routes are
            # correlated to the outputs of all of the routes. Combine them with max.
            if self.crossover[i]:
                # The max function returns a tuple of the max and the indices. We don't need the
                # indices.
                mask_a = mask_b = torch.cat((mask_a, mask_b), dim=1).max(dim=1, keepdim=True)[0]
            # Keep the maximum value of the mask at 1
            mask_max = max(mask_a.max(), mask_b.max())
            mask_a = mask_a / mask_max
            mask_b = mask_b / mask_max

        # Square root for better visualization.
        mask = torch.cat((mask_a, mask_b), dim=1).max(dim=1, keepdim=True)[0].sqrt()

        x = self.classifier(x)
        return x, mask

