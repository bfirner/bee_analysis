#! /usr/bin/python3

"""
Create datasets with basic shapes (square, triangle, chevron, etc).
"""

import argparse
import io
#import math
#import numpy
import os
import random
#import sys
import torch
import webdataset as wds
# Helper function to convert to images
from torchvision import transforms

# A collection of shapes that have some similarities to one another, but should still be simple
# enough to disambiguate.
all_shapes = {
    'square4x4': torch.ones(4,4),
    'square5x5': torch.ones(5,5),
    'square9x9': torch.ones(9,9),
    'triangle3up': torch.tensor([[0, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1]]),
    'triangle3down': torch.tensor([[1, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 0, 0]]),
    'triangle5right': torch.tensor([[0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 1],
                                  [0, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1]]),
    'triangle5left': torch.tensor([[1, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0],
                                 [1, 1, 1, 0, 0],
                                 [1, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1]]),
    'chevron5': torch.tensor([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 0, 0, 0, 1]]),
    'chevron5inverted': torch.tensor([[1, 0, 0, 0, 1],
                                    [1, 1, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 0, 0]]),
    'diamond5': torch.tensor([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]]),
}

parser = argparse.ArgumentParser(
    description="Perform data preparation to create synthetic images for DNN training on a video set.")
parser.add_argument(
    'outpath',
    type=str,
    help='Output directory for the prepared WebDataset.')
parser.add_argument(
    '--width',
    type=int,
    required=False,
    default=224,
    help='Width of output images.')
parser.add_argument(
    '--height',
    type=int,
    required=False,
    default=224,
    help='Height of output images.')
parser.add_argument(
    '--motion',
    type=int,
    required=False,
    default=1,
    help='Pixels of motion between frames.')
parser.add_argument(
    '--frames_per_sample',
    type=int,
    required=False,
    default=1,
    choices=[1],
    help='Number of frames in each sample (only 1 is currently supported).')
parser.add_argument(
    '--shapes_per_sample',
    type=int,
    required=False,
    default=1,
    choices=[1 + x for x in range(len(all_shapes))],
    help='Number of frames in each sample (only 1 is currently supported).')
parser.add_argument(
    '--samples',
    type=int,
    required=False,
    default=5000,
    help='Number of samples in the dataset.')

args = parser.parse_args()


class DataCreator:

    def __init__(self, num_samples, frames_per_sample, shapes_per_sample,
            width, height, speed):
        """
        Samples have no overlaps. For example, a 10 second video at 30fps has 300 samples of 1
        frame, 150 samples of 2 frames with a frame interval of 0, or 100 samples of 2 frames with a
        frame interval of 1.
        Arguments:
            num_samples       (int): Number of generated scenes.
            frames_per_sample (int): Number of frames per scene.
            shapes_per_sample (int): Number of distinct shapes per sample.
            width             (int): Width of output images, or the original width if None.
            height            (int): Height of output images, or the original height if None.
            speed             (int): Pixels of motion between each frame in a sample.
        """
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample
        self.shapes_per_sample = shapes_per_sample
        self.width = width
        self.height = height
        self.speed = speed


    def setSeed(self, seed):
        """Set the seed used for sample generation in the iterator."""
        self.seed = seed


    def __iter__(self):
        """An iterator that yields frames.

        If deterministic behavior is desired then call setSeed before iteration.

        Returns (with each iteration):
            (image, groundtruth)
        """
        # Use a single buffer for all generated frames
        buffer = torch.zeros(self.frames_per_sample, self.height, self.width)
        # Generate all required samples.
        for sample in range(self.num_samples):
            # Pick random shapes from len(shapes) and assign them to random locations
            # Shapes are chosen without replacement, so there is at most one of each shape.
            # The ground truth are the coordinates of all placed shapes, with another output
            # indicating if the shape is present.
            shape_list = list(all_shapes.values())
            selections = random.sample(range(len(all_shapes)), k=self.shapes_per_sample)
            # Create the ground truth. First make some zero tensors then fill in locations and
            # class detection information.
            gt_detection = torch.zeros(len(all_shapes))
            gt_location = torch.zeros(len(all_shapes) * 2)
            for idx in selections:
                gt_detection[idx] = 1
                gt_location[idx*2] = random.randint(0, self.height - shape_list[idx].size(0))
                gt_location[idx*2 + 1] = random.randint(0, self.width - shape_list[idx].size(1))
            groundtruth = (gt_detection, gt_location)
            # TODO Also generate a direction and speed.
            # TODO On subsequent frames move the image.

            # For each frame in frames_per_sample place the shapes in their proper locations.
            for frame in range(self.frames_per_sample):
                # Generate a partial sample using torchvision.transforms.ToPILImage
                for idx in selections:
                    shape = shape_list[idx]
                    y = int(gt_location[idx*2].item())
                    x = int(gt_location[idx*2 + 1].item())
                    buffer[:, y:y+shape.size(0), x:x+shape.size(1)] = shape
            # Yield the sample
            yield buffer, groundtruth

            # Clear the buffer and process the next sample
            buffer.fill_(0)
        return


# Create a writer for the WebDataset
datawriter = wds.TarWriter(args.outpath, encoder=False)

sampler = DataCreator(
    args.samples, args.frames_per_sample, args.shapes_per_sample, args.width, args.height, args.motion)

for sample_num, data in enumerate(sampler):
    frame, ground_truth = data
    height, width = frame.size(1), frame.size(2)
    # If you would like to debug (and you would like to!) check your images.
    if 1 == args.frames_per_sample:
        img = transforms.ToPILImage()(frame[0]).convert('L')
        # Now save the image as a png into a buffer in memory
        buf = io.BytesIO()
        img.save(fp=buf, format="png")

        sample = {
            "__key__": str(sample_num),
            "0.png": buf.getbuffer(),
            # ".pth" is the extension for pytorch data. It just creates a BytesIO object and uses
            # torch.save.
            "detection.pth": wds.torch_dumps(ground_truth[0]),
            "locations.pth": wds.torch_dumps(ground_truth[1]),
        }
    else:
        # Save multiple pngs
        buffers = []

        for i in range(args.frames_per_sample):
            img = transforms.ToPILImage()(frame[i]).convert('L')
            # Now save the image as a png into a buffer in memory
            buffers.append(io.BytesIO())
            img.save(fp=buffers[-1], format="png")

        sample = {
            "__key__": str(sample_num),
            # ".pth" is the extension for pytorch data. It just creates a BytesIO object and uses
            # torch.save.
            "detection.pth": wds.torch_dumps(ground_truth[0]),
            "locations.pth": wds.torch_dumps(ground_truth[1]),
        }
        for i in range(args.frames_per_sample):
            sample[f"{i}.png"] = buffers[i].getbuffer()

    datawriter.write(sample)

datawriter.close()
