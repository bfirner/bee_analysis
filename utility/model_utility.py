#!/usr/bin/python3

import random
import numpy
import torch

from ..models.alexnet import AlexLikeNet
from ..models.bennet import BenNet
from ..models.resnet import (ResNet18, ResNet34)
from ..models.resnext import (ResNext18, ResNext34, ResNext50)
from ..models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)


def createModel(model_type, in_frames, frame_height, frame_width, vector_input_size, output_size):
    """Create a model of the given type. Returns None on failure."""
    net = None
    if 'alexnet' == model_type:
        net = AlexLikeNet(in_dimensions=(in_frames, frame_height, frame_width),
                out_classes=output_size, linear_size=512, vector_input_size=vector_input_size).cuda()
    elif 'resnet18' == model_type:
        net = ResNet18(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size, expanded_linear=True).cuda()
    elif 'resnet34' == model_type:
        net = ResNet34(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size, expanded_linear=True).cuda()
    elif 'bennet' == model_type:
        net = BenNet(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'resnext50' == model_type:
        net = ResNext50(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size, expanded_linear=True).cuda()
    elif 'resnext34' == model_type:
        net = ResNext34(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size, expanded_linear=False,
                use_dropout=False).cuda()
    elif 'resnext18' == model_type:
        net = ResNext18(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size, expanded_linear=True,
                use_dropout=False).cuda()
    elif 'convnextxt' == model_type:
        net = ConvNextExtraTiny(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'convnextt' == model_type:
        net = ConvNextTiny(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'convnexts' == model_type:
        net = ConvNextSmall(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'convnextb' == model_type:
        net = ConvNextBase(in_dimensions=(in_frames, frame_height, frame_width), out_classes=output_size).cuda()
    return net


def restoreModel(resume_from, net):
    """Restore a trained model"""
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint["model_dict"])


def restoreModelAndState(resume_from, net, optimizer):
    """Restore model and optimizer states and RNGs"""
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])
    # Also restore the RNG states
    random.setstate(checkpoint["py_random_state"])
    numpy.random.set_state(checkpoint["np_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

