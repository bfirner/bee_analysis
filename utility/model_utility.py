"""
Copyright © 2025 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

Utilities for creating and reloading models.
"""

import random
import numpy
import os
# TODO It is possible that we may want the onnx-dependent code separate from
# the torch dependent code so we can run on a system without torch.
import onnx
import onnxruntime as ort
import torch

from models.alexnet import AlexLikeNet
from models.bennet import BenNet, CompactingBenNet
from models.dragonfly import DFNet
from models.modules import (Denormalizer, Normalizer)
from models.resnet import (ResNet18, ResNet34)
from models.resnext import (ResNext18, ResNext34, ResNext50)
from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)
# Take NOTE: If the onnx runtime is desired on a platform without torch, pilToNumpy will need to be pulled out of train_utility.
from utility.train_utility import pilToNumpy


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


def restoreModel(resume_from, net, device=torch.device("cpu")):
    """Restore a trained model"""
    checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
    net.load_state_dict(state_dict=checkpoint["model_dict"], strict=True)


def restoreModelAndState(resume_from, net, optimizer):
    """Restore model and optimizer states and RNGs"""
    checkpoint = torch.load(resume_from, weights_only=False)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])
    # Also restore the RNG states
    random.setstate(checkpoint["py_random_state"])
    numpy.random.set_state(checkpoint["np_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])


def hasNormalizers(resume_from) -> bool:
    """Restore a trained model"""
    if torch.backends.cuda.is_built():
        checkpoint = torch.load(resume_from, weights_only=False)
    else:
        checkpoint = torch.load(resume_from, map_location=torch.device('cpu'))
    return checkpoint["denormalizer_state_dict"] is not None and checkpoint["normalizer_state_dict"] is not None


def restoreNormalizers(resume_from, device=None):
    """Restore normalizer and denormalizer. Check with hasNormalizers first."""
    if (device is None and torch.backends.cuda.is_built()) or device == 'cuda':
        checkpoint = torch.load(resume_from, weights_only=False)
    else:
        checkpoint = torch.load(resume_from, map_location=torch.device('cpu'))
    nsd = checkpoint["normalizer_state_dict"]
    dsd = checkpoint["denormalizer_state_dict"]
    normalizer = Normalizer(means=nsd['means'], stddevs=1.0/nsd['inv_stddevs'])
    denormalizer = Denormalizer(means=dsd['means'], stddevs=dsd['stddevs'])
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"], strict=True)
    denormalizer.load_state_dict(checkpoint["denormalizer_state_dict"], strict=True)
    return (
        normalizer, denormalizer
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


def loadModel(model_path, improc={}):
    """
    improc could contain keys for image processing, or they could be reloaded from the model.
    Arguments:
        model_path (str): Path to the model checkpoint
        improc    (dict): Dictionary of metadata and image processing keys
                          The keys are width, height, crop_x_offset, and crop_y_offset.
    Returns:
        (ModelReloader | OnnxReloader)
    """
    if os.path.splitext(model_path)[1] == ".onnx":
        # Create the model
        model = OnnxReloader(model_path, **improc)
    else:
        model = ModelReloader(model_path, **improc)
    return model


class ModelReloader():
    def __init__(self, model_path, scale=None, width=None, height=None,
                 crop_y_offset=None, crop_x_offset=None, half=False, force_cpu=False):
        # Restore the trained model
        if not force_cpu and torch.backends.cuda.is_built():
            self.checkpoint = torch.load(model_path, weights_only=False)
        else:
            self.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.net = createModel2(self.checkpoint["metadata"]["modeltype"], self.checkpoint["metadata"]["model_args"])
        self.net.load_state_dict(state_dict=self.checkpoint["model_dict"], strict=True)
        self.net.eval()
        self.half = half
        if self.half:
            self.net = self.net.half()

        # Initialize patch information from the checkpoint unless it is overwritten
        self.scale = scale or self.checkpoint["metadata"]["improc"]["scale"]
        self.width = width or self.checkpoint["metadata"]["improc"]["width"]
        self.height = height or self.checkpoint["metadata"]["improc"]["height"]
        if crop_y_offset is None:
            self.crop_y_offset = self.checkpoint["metadata"]["improc"]["crop_y_offset"]
        else:
            self.crop_y_offset = crop_y_offset
        if crop_x_offset is None:
            self.crop_x_offset = self.checkpoint["metadata"]["improc"]["crop_x_offset"]
        else:
            self.crop_x_offset = crop_x_offset
        # Other code may use the model's metadata, so keep it up to date
        self.checkpoint["metadata"]["improc"] = {}
        self.checkpoint["metadata"]["improc"]['scale'] = self.scale
        self.checkpoint["metadata"]["improc"]['patch_width'] = self.width
        self.checkpoint["metadata"]["improc"]['patch_height'] = self.height
        self.checkpoint["metadata"]["improc"]['crop_x_offset'] = self.crop_x_offset
        self.checkpoint["metadata"]["improc"]['crop_y_offset'] = self.crop_y_offset

        # Select cpu or cuda backend
        if not force_cpu and torch.backends.cuda.is_built():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.net.to(self.device)
        # Restore the denormalization network, if it was used.
        if hasNormalizers(model_path):
            _, denormalizer = restoreNormalizers(model_path, device=self.device)
            self.denormalizer = denormalizer
            self.denormalizer.eval().to(self.device)
            if self.half:
                self.denormalizer = self.denormalizer.half()
        else:
            self.denormalizer = None

    def imagePreprocess(self, image):
        planes = self.checkpoint['metadata']['model_args']['in_dimensions'][0]
        return inference_common.imagePreprocess(image, self.scale, self.width, self.height, self.crop_x_offset, self.crop_y_offset, planes)

    def imagePreprocessFromCoords(self, image, scale_w, scale_h, crop_coords, src='BGR'):
        planes = self.checkpoint['metadata']['model_args']['in_dimensions'][0]
        return inference_common.imagePreprocessFromCoords(image, scale_w, scale_h, crop_coords, planes=planes, src=src)

    def inferFromPilMemory(self, image):
        orig_w, orig_h = image.size
        scale_w = round(self.scale*orig_w)
        scale_h = round(self.scale*orig_h)
        crop_left = round(scale_w/2 - self.width/2 + self.crop_x_offset)
        crop_top = round(scale_h/2 - self.height/2 + self.crop_y_offset)
        crop_coords = (crop_left, crop_top, crop_left + self.width, crop_top + self.height)

        image = image.resize((scale_w, scale_h)).crop(crop_coords)
        image = torch.tensor(pilToNumpy(image)).to(self.device)

        with torch.no_grad():
            if self.checkpoint['metadata']['normalize_images']:
                image = normalizeImages(image)

            if self.half:
                output = self.net.forward(image.half())
            else:
                output = self.net.forward(image)
            if self.denormalizer is not None:
                output = self.denormalizer(output)[0].tolist()
            else:
                output = output[0].tolist()
            return output

    def inferFromPil(self, path):
        if self.checkpoint['metadata']['model_args']['in_dimensions'][0] == 1:
            with Image.open(os.path.join(path)).convert("L") as image:
                return inferFromPilMemory(image)
        else:
            with Image.open(os.path.join(path)).convert("RGB") as image:
                return inferFromPilMemory(image)

    def infer(self, image):
        # TODO FIXME Add an option to infer with a status input
        image = self.imagePreprocess(image)
        return self.inferPatch(image)

    def inferPatch(self, patch):
        patch = torch.tensor(pilToNumpy(patch)).to(self.device)
        with torch.no_grad():
            if self.checkpoint['metadata']['normalize_images']:
                patch = normalizeImages(patch)
            if self.half:
                output = self.net.forward(patch.half())
            else:
                output = self.net.forward(patch)
            if self.denormalizer is not None:
                output = self.denormalizer(output)[0].tolist()
            else:
                output = output[0].tolist()
            return output


# TODO This can live with the OnnxReloader for now, in case that code is moved into file without torch dependence.
def normalizeImagesNumpy(images, epsilon=1e-05):
    # normalize per channel, so compute over height and width. This handles images with or without a batch dimension.
    v = numpy.var(images, axis=(images.ndim - 2, images.ndim - 1), keepdims=True)
    m = numpy.mean(images, axis=(images.ndim - 2, images.ndim - 1), keepdims=True)
    return (images - m) / (v + epsilon)


class OnnxReloader():
    """Reload a model from a saved compute graph."""
    def __init__(self, model_path, improc, dimensions=3, normalize=True):
        """
        Arguments:
            model_path  (str): Path to the onnx graph
            improc     (dict): image processing information (as from dataprep config)
            dimensions  (int): Number of input planes
            normalize  (bool): True if input images should be normalized
        """
        # Restore the traced graph
        self.net = onnx.load(model_path)

        onnx.checker.check_model(self.net)

        self.inference = ort.InferenceSession(model_path)

        # TODO Needs a model name or this barfs
        #print(onnx.helper.printable_graph(self.net))

        # Initialize patch information from the checkpoint unless it is overwritten
        self.scale = improc["scale"]
        self.width = improc['width']
        self.height = improc['height']
        self.crop_x_offset = improc['crop_x_offset']
        self.crop_y_offset = improc['crop_y_offset']
        self.dimensions = dimensions
        self.normalize = normalize

        # TODO FIXME Load this from somewhere
        self.checkpoint = {
            'metadata': {
                'labels': ["pixeltarget_x", "pixeltarget_y", "pixeltarget_radius", "pixeltarget_present"],
                'in_dimensions': dimensions,
            }
        }

    def imagePreprocess(self, image):
        return inference_common.imagePreprocess(image, self.scale, self.width, self.height, self.crop_x_offset, self.crop_y_offset, self.dimensions)

    def imagePreprocessFromCoords(self, image, scale_w, scale_h, crop_coords, src='BGR'):
        return inference_common.imagePreprocessFromCoords(image, scale_w, scale_h, crop_coords, planes=self.dimensions, src=src)

    def inferFromPilMemory(self, image):
        orig_w, orig_h = image.size
        scale_w = round(self.scale*orig_w)
        scale_h = round(self.scale*orig_h)
        crop_left = round(scale_w/2 - self.width/2 + self.crop_x_offset)
        crop_top = round(scale_h/2 - self.height/2 + self.crop_y_offset)
        crop_coords = (crop_left, crop_top, crop_left + self.width, crop_top + self.height)

        image = image.resize((scale_w, scale_h)).crop(crop_coords)
        patch = pilToNumpy(image)

        if self.normalize:
            patch = normalizeImagesNumpy(images)
        return self.inference.run(None, {'patch': patch})[0].tolist()[0]

    def inferFromPil(self, path):
        if self.checkpoint['metadata']['model_args']['in_dimensions'][0] == 1:
            with Image.open(os.path.join(path)).convert("L") as image:
                return inferFromPilMemory(image)
        else:
            with Image.open(os.path.join(path)).convert("RGB") as image:
                return inferFromPilMemory(image)

    def infer(self, image):
        # TODO FIXME Add an option to infer with a status input
        image = self.imagePreprocess(image)
        return self.inferPatch(image)

    def inferPatch(self, patch):
        patch = pilToNumpy(patch)
        if self.normalize:
            patch = normalizeImagesNumpy(patch)
        return self.inference.run(None, {'patch': patch})[0].tolist()[0]
