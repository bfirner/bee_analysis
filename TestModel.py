#! /usr/bin/python3

"""
Copyright © 2025 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

Load a model and test it on a dataset or video.
Crop parameters will be read from the model file or can be provided in a yaml file with the following fields:
crop_x_offset: 341
crop_y_offset: 372
height: 210
width: 1086
frames_per_sample: 1
scale: 1.0

Channels will be deduced from the model file and the frames_per_sample.
"""

import argparse
import numpy
import csv
import os
import sys
import time
import torch
import yaml
from torchvision.transforms import v2 as transforms


from models.modules import (Denormalizer, Normalizer)
from utility.model_utility import (loadModel)
from utility.train_utility import (normalizeImages)
#from utility.video_utility import (getVideoInfo, VideoSampler)
from utility.frame_provider import (VideoReader)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    required=True,
    type=str,
    help="Filename of the model file (.pyt or .onnx) to use.")
parser.add_argument(
    '--improc_config',
    required=False,
    type=str,
    help=("Filename of the image processing yaml file.",
          " If not provided, settings will come from the model file."))
parser.add_argument(
    '--logfile',
    required=True,
    type=str,
    help="Filename to use for saving results.")
parser.add_argument(
    'paths',
    type=str,
    nargs='+',
    help="A list of videos for testing.")
parser.add_argument(
    '--background_subtraction',
    type=str,
    required=False,
    choices=['none', 'mog2', 'knn'],
    default='none',
    help='Background subtraction algorithm to apply to the input video, or none.')
parser.add_argument(
    '--channels',
    required=False,
    type=int,
    default=1,
    help="The expected number of channels for images given to the model.")
parser.add_argument(
    '--frame_interval',
    required=False,
    type=int,
    default=1,
    help="The frame interval for multiframe inputs.")

args = parser.parse_args()

##########
# Load the image processing config, if provided.
improc = {}
if args.improc_config is not None:
    with open(args.improc_config, "r", newline=None) as improc_file:
        improc = yaml.safe_load(improc_file)

##########
# Reload the model.

model = loadModel(args.model, improc)
frames_per_sample = model.checkpoint['metadata']['model_args']['in_dimensions'][0]//args.channels

##########
# Get the image processing config from the model or provided config file.

if 3 == args.channels:
    data_to_pil = lambda frame: transforms.ToPILImage()(frame/255.0).convert('RGB')
else:
    data_to_pil = lambda frame: transforms.ToPILImage()(frame/255.0).convert('L')

##########
# Load each video

buffer_len = args.frames_per_sample * (args.interval - 1)
if buffer_len > 0:
    frame_buffer = 0

for path in args.paths:
    #vid_width, vid_height, vid_frames = getVideoInfo(path)
    ## TODO FIXME args.samples should be all the frames, args.frames_per_sample comes from model.checkpoint['metadata']['model_args']['in_dimensions'][0]//args.channels, and args.interval must be provided. Should probably be in the improc file.
    #sampler = VideoSampler(
    #    video_path=path, num_samples=vid_frames, frames_per_sample=frames_per_sample,
    #    frame_interval=args.frame_interval, out_width=model.width, out_height=model.height,
    #    crop_noise=0, scale=model.scale, crop_x_offset=model.crop_x_offset,
    #    crop_y_offset=model.crop_y_offset, channels=args.channels,
    #    begin_frame=0, end_frame=vid_frames-1, normalize=False,
    #    bg_subtract=args.background_subtraction)

    reader = VideoReader(path=path)
    vid_width, vid_height = reader.imageSize()
    vid_frames = reader.totalFrames()

    ##########
    # Iterate through the data and print out results
    with open(args.logfile, "w", newline=None) as log:
        for frame_num, frame in enumerate(reader):
        #for sample_num, frame_data in enumerate(sampler):
            frame, video_path, frame_num = frame_data
            base_name = os.path.basename(video_path).replace(' ', '_').replace('.', '_')
            video_time = os.path.basename(video_path).split('.')[0]
            # TODO FIXME Convert the time from the video to the current frame time.
            # TODO Assuming 3fps bee videos
            time_sec = time.mktime(time.strptime(video_time, "%Y-%m-%d %H:%M:%S"))
            time_struct = time.localtime(time_sec + int(frame_num[0]) // 3)
            curtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
            metadata = f"{video_path},{frame_num[0]},{curtime}"
            height, width = frame.size(2), frame.size(3)

            # Convert to tensors and forward through the model
            if 1 == frames_per_sample:
                img = data_to_pil(frame[0])
                img_input = img
            else:
                imgs = [data_to_pil(frame[i]) for i in range(frames_per_sample)]
                img_input = torch.cat(img, dim=0)

            # This applies all pre- and post- processing
            output = model.inferFromPilMemory(img_input)

            # Write the result
            log.write(f"{sample_num}:  output")

            # Update the buffer
            if buffer_len > 0:
                frame_buffer.push(frame)
                frame_buffer.pop()
