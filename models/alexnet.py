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

        # Initialize in a no_grad section so that we can fill in some initial values for the bias
        # tensors.
        with torch.no_grad():
            for i in range(len(self.kernels)):
                layer, out_size = self.createConvLayer(i, out_size)
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
                nn.Softmax(dim=1)
            )
            self.classifier[0].bias.fill_(1.)

    #TODO AMP
    #@autocast()
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



