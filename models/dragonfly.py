"""
This is a model that has layers that emulate some of the tricks in the dragonfly visual system.
See _Feature-Detecting Neurons in Dragonflies_: https://www.nature.com/articles/362541a0
And _Visual Receptive Field Properties of Feature Detecting Neurons in the Dragonfly_: https://link.springer.com/article/10.1007/BF00207186
"""

import itertools
import math
import torch
import torch.nn as nn
import torchvision.ops as ops

from torchvision import disable_beta_transforms_warning
from torchvision.transforms import v2 as transforms
from torchvision.transforms import InterpolationMode 

from models.modules import MaxThresholding, PresolvedConv2d, PresolvedLinear


class DFNet(nn.Module):
    """A small dragonfly network."""

    def initSmallSquareFilters(self):
        """Initialize small square (5x5 perceptive field) filters."""
        # Go from self.channels[0] to self.channels[1]
        kernel_size = 3
        return torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[0], out_channels=self.channels[1],
                kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1], out_channels=self.channels[1],
                kernel_size=kernel_size, stride=self.strides[0], padding=kernel_size//2,
                groups=self.channels[1]//16),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
        )

    def initHorizontalBarFilters(self):
        """Initialize small horizontal bar (1x9 perceptive field) filters."""
        # Go from self.channels[0] to self.channels[1]
        kernel_width = 5
        return torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[0], out_channels=self.channels[1],
                kernel_size=(1, kernel_width), stride=1, padding=(0, kernel_width//2)),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1], out_channels=self.channels[1],
                kernel_size=(1, kernel_width), stride=self.strides[0], padding=(0, kernel_width//2),
                groups=self.channels[1]//16),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
        )

    def initVerticalBarFilters(self):
        """Initialize small vertical bar (9x1 perceptive field) filters."""
        # Go from self.channels[0] to self.channels[1]
        kernel_height = 5
        return torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[0], out_channels=self.channels[1],
                kernel_size=(kernel_height, 1), stride=1, padding=(kernel_height//2, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1], out_channels=self.channels[1],
                kernel_size=(kernel_height, 1), stride=self.strides[0], padding=(kernel_height//2, 0),
                groups=self.channels[1]//16),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
        )

    def initLargeSquareFilters(self):
        """Initialize large square (13x13 perceptive field) filters."""
        # Go from self.channels[0] to self.channels[1]
        kernel_size = 7
        return torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[0], out_channels=self.channels[1],
                kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1], out_channels=self.channels[1],
                kernel_size=kernel_size, stride=self.strides[0], padding=kernel_size//2,
                groups=self.channels[1]//16),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[1]),
        )

    def initLocalFilters(self):
        """Initialize two regular convolution filters."""
        # Go from 4*self.channels[1] to self.channels[2] and self.channels[3]
        kernel_size = 3
        return torch.nn.Sequential(
            nn.Conv2d(
                in_channels=4*self.channels[1], out_channels=self.channels[2],
                kernel_size=kernel_size, stride=self.strides[1], padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2], out_channels=self.channels[3],
                kernel_size=kernel_size, stride=self.strides[2], padding=kernel_size//2),
            nn.ReLU(),
            # TODO Getting rid of this may improve local generalization/decrease local fit
            nn.BatchNorm2d(self.channels[3]),
        )

    def initCompactingFilters(self):
        """Initialize compacting filters that reduce channel dimensions to 1x1."""
        # Go from self.channels[3] to self.channels[4]
        height = self.layer_sizes[-1][0]
        width = self.layer_sizes[-1][1]
        return torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[3], out_channels=self.channels[4],
                kernel_size=(1, width), stride=1, groups=self.channels[3]//16),
            nn.Conv2d(
                in_channels=self.channels[4], out_channels=self.channels[4],
                kernel_size=(height, 1), stride=1, groups=self.channels[4]//16),
            nn.Flatten(),
        )

    def __init__(self, in_dimensions, out_classes, vector_input_size=0):
        """
        Arguments:
            in_dimensions (tuple(int)): Tuple of channels, height, and width.
            out_classes          (int): The number of output classes.
            vector_input_size    (int): The number of vector inputs to the linear layers.
        """
        super(DFNet, self).__init__()

        # Disable some warnings about the rotations being in beta
        disable_beta_transforms_warning()

        in_dimensions = list(in_dimensions)

        # The network will forward the image and two of its rotations
        self.in_dimensions = [in_dimensions[0]*3] + in_dimensions[1:]
        self.out_classes = out_classes

        # Call the overridable function that initializes internal settings
        self.initializeSizes()

        # Features are extracted with four sets of filters:
        # 1. Horizontal bar filters
        # 2. Vertical bar filters
        # 3. Small square filters
        # 4. Large square filters
        self.local_filters = torch.nn.ModuleDict({
            'small_squares': self.initSmallSquareFilters(),
            'horizontal_bars': self.initHorizontalBarFilters(),
            'vertical_bars': self.initVerticalBarFilters(),
            'large_squares': self.initLargeSquareFilters(),
        })

        # Outputs from the first three sets of filters are set through a local supression layer.
        # This layer produces a sparse output by running a filter that only emits the local maximum.
        # For example, if a texture is detected across most of an image the channel input to this
        # layer would be a channel mostly filled with the magnitude of that feature, m, and the
        # noise response in the rest of the image, n. A kernel is run over the input channel with
        # negative values in all locations except for the center, and the output is then passed
        # through a ReLU. This supresses all responses except at the edges of the texture, basically
        # producing an outline of the feature that has that texture.
        self.suppression_activation = nn.ReLU()
        weight_list = (3 * self.channels[1]) * [[[[-0.2]*3,  [-0.2, 1., -0.2], [-0.2]*3]]]
        self.suppression_weight = nn.parameter.Parameter(data=torch.tensor(weight_list), requires_grad=False)

        # The output of the local suppression layer and the large square kernels are combined and
        # passed through a squeeze excitation layer
        self.squeeze_excite = ops.SqueezeExcitation(
            input_channels = self.channels[1] * 4,
            # The internal channels were set to 1/16 of the input channels in the original paper,
            # where the reduction ratio was referred to as 'r':
            # "Squeeze-and-Excitation Networks" (https://arxiv.org/pdf/1709.01507.pdf)
            squeeze_channels = self.channels[1]//4,
            activation = torch.nn.ReLU,
            scale_activation = torch.nn.Sigmoid)

        # Two local convolution are used to downscale the layer
        self.local_convs = self.initLocalFilters()

        # The features are then compacted with large horizontal and vertical bar filters
        self.compactor = self.initCompactingFilters()

        # Linear layers accept the flattened feature maps.
        linear_input_size = self.channels[-1] + vector_input_size
        self.classifier = nn.Sequential(
            nn.Linear(in_features=linear_input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=96),
            nn.ReLU(),
            nn.Linear(in_features=96, out_features=self.out_classes)
            # No softmax at the end. To train a single label classifier use CrossEntropyLoss
            # rather than NLLLoss. This allows for multi-label classifiers trained with BCELoss.
        )


    def initializeSizes(self):
        """
        Override this function to change internal layer parameters.
        """
        # The first channel number after the input dimensions are for the four filter banks.
        # Next is the channel counts for two local convolutions that follow the squeeze excitation
        # layer.
        # Finally, the channel size of the compacting convolution.
        # Kernel sizes always have odd dimensions and padding is always kernel_size//2.
        # TODO Because of grouping, use multiples of G, where G is some grouping size (testing # grouping is a TODO)
        self.channels = (self.in_dimensions[0], 16, 64, 128, 256)
        self.strides = (2, 2, 2, 1)

        assert(len(self.strides) == (len(self.channels) - 1))

        # From the in_dimensions, channels, and strides we can calculate the sizes of everything
        # that follows.
        self.layer_sizes = [self.in_dimensions[1:]]
        for layer in range(1, len(self.channels)):
            out_size = (int((self.layer_sizes[layer-1][0] - 1)/self.strides[layer-1] + 1),
                        int((self.layer_sizes[layer-1][1] - 1)/self.strides[layer-1] + 1))
            self.layer_sizes.append(out_size)


    def forward(self, x, vector_input = None):
        """Forward pass."""
        features = self.forwardToFeatures(x)

        if vector_input is not None:
            features = torch.cat((features, vector_input), dim=1)

        results = self.classifier(features)
        return results

    def forwardToFeatures(self, x):
        """Produce and return the feature maps before the linear layers."""
        # Do the initial rotations so that there are three versions of the image
        img45 = transforms.functional.rotate(inpt=x, angle=45, interpolation=InterpolationMode.BILINEAR, expand=False)
        img90 = transforms.functional.rotate(inpt=x, angle=90, interpolation=InterpolationMode.BILINEAR, expand=False)

        all_imgs = torch.cat((x, img45, img90), dim=1)

        # Send the images through all of the local filters
        x_ss = self.local_filters['small_squares'](all_imgs)
        x_hb = self.local_filters['horizontal_bars'](all_imgs)
        x_vb = self.local_filters['vertical_bars'](all_imgs)
        x_ls = self.local_filters['large_squares'](all_imgs)

        # The outputs of the first three filters go through local suppression
        suppressed_features = self.suppression_activation(
            torch.nn.functional.conv2d(
                input=torch.cat((x_ss, x_hb, x_vb), dim=1),
                weight=self.suppression_weight.data,
                bias=None, stride=1, padding=1, dilation=1, groups=3 * self.channels[1]))

        # Squeeze excitation
        x = torch.cat((suppressed_features, x_ls), dim=1)
        x = self.squeeze_excite(x)

        # Local convolutions
        x = self.local_convs(x)

        # Compaction and flattening
        x = self.compactor(x)

        return x



