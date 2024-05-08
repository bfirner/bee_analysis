#! /usr/bin/python3

"""
This program will prepare video data for DNN training ("dataprep").
A csv will provide a video list to load and metainformation and/or labels used for training.
This program will perform frame extraction, augmentations, and other specified processed and then
save the video data and labels for training or validation.
"""

# Not using video reading library from torchvision.
# It only works with old versions of ffmpeg.
import argparse
import csv
import ffmpeg
import io
import math
import numpy
import os
import random
import sys
import time
import torch
import webdataset as wds
# Helper function to convert to images
from torchvision import transforms

from utility.video_utility import (getVideoInfo, VideoSampler, vidSamplingCommonCrop)


parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set.")
parser.add_argument(
    'datalist',
    type=str,
    help=('A csv file with video files and labels.'
          ' Columns should be "file," "label," "begin," and "end".'
          ' The "begin" and "end" columns are in frames.'
          ' If the "begin" and "end" columns are not present,'
          ' then the label will be applied to the entire video.'))
parser.add_argument(
    'outpath',
    type=str,
    help='Output directory for the prepared WebDataset.')
parser.add_argument(
    '--width',
    type=int,
    required=False,
    default=224,
    help='Width of output images (obtained via cropping, after applying scale).')
parser.add_argument(
    '--height',
    type=int,
    required=False,
    default=224,
    help='Height of output images (obtained via cropping, after applying scale).')
parser.add_argument(
    '--resize-strategy',
    type=str,
    required=False,
    default='crop',
    choices=['crop', 'scale'],
    help='This deprecated option is ignored. Use --scale to scale, and crop with --width and '
    '--height, with --crop_noise and --crop_x_offset and --crop_y_offset for more options.')
parser.add_argument(
    '--scale',
    type=float,
    required=False,
    default=1.0,
    help='Scaling to apply to each dimension (before cropping). A value of 0.5 will yield 0.25 resolution.')
parser.add_argument(
    '--crop_noise',
    type=int,
    required=False,
    default=0,
    help='The noise (in pixels) to randomly add to the crop location in both the x and y axis.')
parser.add_argument(
    '--crop_x_offset',
    type=int,
    required=False,
    default=0,
    help='The offset (in pixels) of the crop location on the scaled image in the x dimension.')
parser.add_argument(
    '--crop_y_offset',
    type=int,
    required=False,
    default=0,
    help='The offset (in pixels) of the crop location on the scaled image in the y dimension.')
parser.add_argument(
    '--interval',
    type=int,
    required=False,
    default=0,
    help='Frames to skip between frames in a sample.')
parser.add_argument(
    '--frames_per_sample',
    type=int,
    required=False,
    default=1,
    help='Number of frames in each sample.')
parser.add_argument(
    '--samples',
    type=int,
    required=False,
    default=5,
    help='Number of samples from each video.')
parser.add_argument(
    '--out_channels',
    type=int,
    required=False,
    choices=[1, 3],
    default=3,
    help='Channels of output images.')
parser.add_argument(
    '--threads',
    type=int,
    required=False,
    default=1,
    help='Number of thread workers')
parser.add_argument(
    '--background_subtraction',
    type=str,
    required=False,
    choices=['none', 'mog2', 'knn'],
    default='none',
    help='Background subtraction algorithm to apply to the input video, or none.')

args = parser.parse_args()

# Create a writer for the WebDataset
datawriter = wds.TarWriter(args.outpath, encoder=False)

with open(args.datalist, newline='') as datacsv:
    conf_reader = csv.reader(datacsv)
    header = next(conf_reader)
    # Remove all spaces from the header strings
    header = [''.join(col.split(' ')) for col in header]
    file_col = header.index('file')
    class_col = header.index('class')
    beginf_col = header.index('beginframe')
    endf_col = header.index('endframe')
    for row in conf_reader:
        # Read the next video
        # Make sure that this line is sane
        if 4 != len(row):
            print(f"Row '{row}' does not have the correct number of columns (4).")
        else:
            path = row[file_col]
            try: 
                sampler = VideoSampler(
                    video_path=path, num_samples=args.samples, frames_per_sample=args.frames_per_sample,
                    frame_interval=args.interval, out_width=args.width, out_height=args.height,
                    crop_noise=args.crop_noise, scale=args.scale, crop_x_offset=args.crop_x_offset,
                    crop_y_offset=args.crop_y_offset, channels=args.out_channels,
                    begin_frame=row[beginf_col], end_frame=row[endf_col],
                    bg_subtract=args.background_subtraction)
            except ffmpeg.Error as e:
                print('stdout:', e.stdout.decode('utf8'))
                print('stderr:', e.stderr.decode('utf8'))
                os.exit(-1)
                
            for sample_num, frame_data in enumerate(sampler):
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
                # Now crop to args.width by args.height.
                #ybegin = (height - args.height)//2
                #xbegin = (width - args.width)//2
                #cropped = frame[:,:,ybegin:ybegin+args.height,xbegin:xbegin+args.width]
                # If you would like to debug (and you would like to!) check your images.
                if 1 == args.frames_per_sample:
                    if 3 == args.out_channels:
                        img = transforms.ToPILImage()(frame[0]/255.0).convert('RGB')
                    else:
                        img = transforms.ToPILImage()(frame[0]/255.0).convert('L')
                    # Now save the image as a png into a buffer in memory
                    buf = io.BytesIO()
                    img.save(fp=buf, format="png")

                    sample = {
                        "__key__": '_'.join((base_name, '_'.join(frame_num))),
                        "0.png": buf.getbuffer(),
                        "cls": row[class_col].encode('utf-8'),
                        "metadata.txt": metadata.encode('utf-8'),
                        "image_scale": str(args.scale).encode('utf-8'),
                        "patch_width": str(args.width).encode('utf-8'),
                        "patch_height": str(args.height).encode('utf-8'),
                        "crop_x_offset": str(args.crop_x_offset).encode('utf-8'),
                        "crop_y_offset": str(args.crop_y_offset).encode('utf-8'),
                        "original_width": str(sampler.width).encode('utf-8'),
                        "original_height": str(sampler.height).encode('utf-8'),
                    }
                else:
                    # Save multiple pngs
                    buffers = []

                    for i in range(args.frames_per_sample):
                        if 3 == args.out_channels:
                            img = transforms.ToPILImage()(frame[i]/255.0).convert('RGB')
                        else:
                            img = transforms.ToPILImage()(frame[i]/255.0).convert('L')
                        # Now save the image as a png into a buffer in memory
                        buffers.append(io.BytesIO())
                        img.save(fp=buffers[-1], format="png")

                    sample = {
                        "__key__": '_'.join((base_name, '_'.join(frame_num))),
                        "cls": row[class_col].encode('utf-8'),
                        "metadata.txt": metadata.encode('utf-8'),
                        "image_scale": str(args.scale).encode('utf-8'),
                        "patch_width": str(args.width).encode('utf-8'),
                        "patch_height": str(args.height).encode('utf-8'),
                        "crop_x_offset": str(args.crop_x_offset).encode('utf-8'),
                        "crop_y_offset": str(args.crop_y_offset).encode('utf-8'),
                        "original_width": str(sampler.width).encode('utf-8'),
                        "original_height": str(sampler.height).encode('utf-8'),
                    }
                    for i in range(args.frames_per_sample):
                        sample[f"{i}.png"] = buffers[i].getbuffer()

                datawriter.write(sample)

datawriter.close()
