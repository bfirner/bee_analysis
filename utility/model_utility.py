import random
import numpy
import torch

from models.alexnet import AlexLikeNet
from models.bennet import BenNet, CompactingBenNet
from models.dragonfly import DFNet
from models.modules import (Denormalizer, Normalizer)
from models.resnet import (ResNet18, ResNet34)
from models.resnext import (ResNext18, ResNext34, ResNext50)
from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)


def createModel2(model_type, other_args):
    """Create a model of the given type. Returns None on failure."""
    net = None
    if 'alexnet' == model_type:
        net = AlexLikeNet(**other_args).cuda()
    elif 'resnet18' == model_type:
        net = ResNet18(**other_args).cuda()
    elif 'resnet34' == model_type:
        net = ResNet34(**other_args).cuda()
    elif 'bennet' == model_type:
        net = BenNet(**other_args).cuda()
    elif 'compactingbennet' == model_type:
        net = CompactingBenNet(**other_args).cuda()
    elif 'dragonfly' == model_type:
        net = DFNet(**other_args).cuda()
    return net


def createModel(model_type, in_channels, frame_height, frame_width, output_size,
        other_args=[]):
    """Create a model of the given type. Returns None on failure."""
    net = None
    if 'alexnet' == model_type:
        net = AlexLikeNet(in_dimensions=(in_channels, frame_height, frame_width),
                out_classes=output_size, **other_args).cuda()
    elif 'resnet18' == model_type:
        net = ResNet18(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size, **other_args).cuda()
    elif 'resnet34' == model_type:
        net = ResNet34(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size, **other_args).cuda()
    elif 'bennet' == model_type:
        net = BenNet(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size, **other_args).cuda()
    elif 'resnext50' == model_type:
        net = ResNext50(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size, expanded_linear=True).cuda()
    elif 'resnext34' == model_type:
        net = ResNext34(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size, expanded_linear=False,
                use_dropout=False).cuda()
    elif 'resnext18' == model_type:
        net = ResNext18(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size, expanded_linear=True,
                use_dropout=False).cuda()
    elif 'convnextxt' == model_type:
        net = ConvNextExtraTiny(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'convnextt' == model_type:
        net = ConvNextTiny(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'convnexts' == model_type:
        net = ConvNextSmall(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size).cuda()
    elif 'convnextb' == model_type:
        net = ConvNextBase(in_dimensions=(in_channels, frame_height, frame_width), out_classes=output_size).cuda()
    return net


def restoreModel(resume_from, net):
    """Restore a trained model"""
    checkpoint = torch.load(resume_from)
    net.load_state_dict(state_dict=checkpoint["model_dict"], strict=True)


def restoreModelAndState(resume_from, net, optimizer):
    """Restore model and optimizer states and RNGs"""
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])
    # Also restore the RNG states
    random.setstate(checkpoint["py_random_state"])
    numpy.random.set_state(checkpoint["np_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])


def hasNormalizers(resume_from) -> bool:
    """Restore a trained model"""
    checkpoint = torch.load(resume_from)
    return checkpoint["denormalizer_state_dict"] is not None and checkpoint["normalizer_state_dict"] is not None


def restoreNormalizers(resume_from):
    """Restore normalizer and denormalizer. Check with hasNormalizers first."""
    checkpoint = torch.load(resume_from)
    nsd = checkpoint["normalizer_state_dict"]
    dsd = checkpoint["denormalizer_state_dict"]
    return (
        Normalizer(means=nsd['means'], stddevs=1.0/nsd['inv_stddevs']),
        Denormalizer(means=dsd['means'], stddevs=dsd['stddevs']),
    )


def getLabelLocations(metadata):
    """Get the label locations, a list of int and slice values, given a model's metadata."""
    output_locations = {}
    out_idx = 0
    for output_name, output_size in zip(metadata['labels'], metadata['label_sizes']):
        if 1 == output_size:
            output_locations[output_name] = out_idx
        else:
            output_locations[output_name] = slice(out_idx, out_idx + output_size)
        out_idx += output_size
    return output_locations
