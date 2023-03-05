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

from utility.video_utility import (getVideoInfo, vidSamplingCommonCrop)


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
    help='The offset (in pixels) of the crop location on the original image in the x dimension.')
parser.add_argument(
    '--crop_y_offset',
    type=int,
    required=False,
    default=0,
    help='The offset (in pixels) of the crop location on the original image in the y dimension.')
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

class VideoSampler:

    def __init__(self, video_path, num_samples, frames_per_sample, frame_interval,
            out_width=None, out_height=None, crop_noise=0, scale=1.0, crop_x_offset=0,
             crop_y_offset=0, channels=3, begin_frame=None, end_frame=None,
             bg_subtract='none'):
        """
        Samples have no overlaps. For example, a 10 second video at 30fps has 300 samples of 1
        frame, 150 samples of 2 frames with a frame interval of 0, or 100 samples of 2 frames with a
        frame interval of 1.
        Arguments:
            video_path  (str): Path to the video.
            num_samples (int): Number of samples yielded from VideoSampler's iterator.
            frames_per_sample (int):  Number of frames in each sample.
            frame_interval    (int): Number of frames to skip between each sampled frame.
            out_width     (int): Width of output images, or the original width if None.
            out_height    (int): Height of output images, or the original height if None.
            crop_noise    (int): Noise to add to the crop location (in both x and y dimensions)
            scale       (float): Scale factor of each dimension
            crop_x_offset (int): x offset of crop, in pixels, from the original image
            crop_y_offset (int): y offset of crop, in pixels, from the original image
            channels      (int): Numbers of channels (3 for RGB or 1 luminance/Y/grayscale/whatever)
            begin_frame   (int): First frame to possibly sample.
            end_frame     (int): Final frame to possibly sample.
            bg_subtract   (str): Type of background subtraction to use (mog2 or knn), or none.
        """
        self.path = video_path
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample
        self.frame_interval = frame_interval
        self.channels = channels
        self.scale = scale

        # Background subtraction will require openCV if requested.
        self.bg_subtractor = None
        if ('none' != bg_subtract):
            from cv2 import (createBackgroundSubtractorMOG2,
                             createBackgroundSubtractorKNN)
            if 'mog2' == bg_subtract:
                self.bg_subtractor = createBackgroundSubtractorMOG2()
            elif 'knn' == bg_subtract:
                self.bg_subtractor = createBackgroundSubtractorKNN()

        print(f"Processing {video_path}")
        # Probe the video to find out some metainformation

        self.width, self.height, self.total_frames = getVideoInfo(video_path)

        if out_width is None or out_height is None:
            self.crop_noise = 0
        else:
            self.crop_noise = crop_noise

        self.out_width, self.out_height, self.crop_x, self.crop_y = vidSamplingCommonCrop(
            self.height, self.width, out_height, out_width, self.scale, crop_x_offset, crop_y_offset)

        if begin_frame is None:
            self.begin_frame = 1
        else:
            self.begin_frame = int(begin_frame)

        if end_frame is None:
            self.end_frame = self.total_frames
        else:
            # Don't attempt to sample more frames than there exist.
            self.end_frame = min(int(end_frame), self.total_frames)
        # Don't attempt to make more samples that the number of frames that will be sampled.
        # Remember that the frames in frame_interval aren't used but are still skipped along with
        # each sample.
        self.sample_span = self.frames_per_sample + (self.frames_per_sample - 1) * self.frame_interval
        self.available_samples = (self.end_frame - (self.sample_span - 1) - self.begin_frame)//self.sample_span
        self.num_samples = min(self.available_samples, self.num_samples)
        print(f"Video begin and end frames are {self.begin_frame} and {self.end_frame}")
        print(f"Video has {self.available_samples} available samples of size {self.sample_span} and {self.num_samples} will be sampled")


    def setSeed(self, seed):
        """Set the seed used for sample generation in the iterator."""
        self.seed = seed


    def __iter__(self):
        """An iterator that yields frames.

        The entire video will always be decoded and samples will be returned along the way. This
        means that the samples will always be in order. It is assumed that the consumer will shuffle
        them if that behavior is desired. This also means that frames will be sampled without
        replacement. For replacement, just iterate multiple times.
        If deterministic behavior is desired then call setSeed before iteration.

        Returns (with each iteration):
            (image, path, (frames))
        """
        # Determine where frames to sample.
        target_samples = [(self.begin_frame - 1) + x * self.sample_span for x in sorted(random.sample(
            population=range(self.available_samples), k=self.num_samples))]
        # Open the video
        # It is a bit unfortunate the we decode what is probably a YUV stream into rgb24, but this
        # is what PIL supports easily. It is only really detrimental when we want just the Y
        # channel.
        if 3 == self.channels:
            pix_fmt='rgb24'
        else:
            pix_fmt='gray'
        in_width = self.out_width + 2 * self.crop_noise
        in_height = self.out_height + 2 * self.crop_noise

        # Initialize the background subtractor
        if (self.bg_subtractor is not None):
            from cv2 import (bitwise_and)
            # Read in a few hundred frames
            process1 = (
                ffmpeg
                .input(self.path)
                # 400 is the default window for background subtractors
                .trim(start_frame=1, end_frame=400)
                # Scale
                .filter('scale', self.scale*self.width, -1)
                # The crop is automatically centered if the x and y parameters are not used.
                .filter('crop', out_w=in_width, out_h=in_height, x=self.crop_x, y=self.crop_y)
                # Full independence between color channels. The bee videos are basically a single color.
                # Otherwise normalizing the channels independently may not be a good choice.
                .filter('normalize', independence=1.0)
                #.filter('reverse')
                # YUV444p is the alternative to rgb24, but the pretrained network expects rgb images.
                #.output('pipe:', format='rawvideo', pix_fmt='yuv444p')
                .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                .run_async(pipe_stdout=True, quiet=True)
            )
            in_bytes = process1.stdout.read(in_width * in_height * self.channels)
            if in_bytes:
                # Convert to numpy and feed to the background subtraction algorithm
                np_frame = numpy.frombuffer(in_bytes, numpy.uint8)
                fgMask = self.bg_subtractor.apply(np_frame)

        process1 = (
            ffmpeg
            .input(self.path)
            # Scale
            .filter('scale', self.scale*self.width, -1)
            # The crop is automatically centered if the x and y parameters are not used.
            .filter('crop', out_w=in_width, out_h=in_height, x=self.crop_x, y=self.crop_y)
            # Full independence between color channels. The bee videos are basically a single color.
            # Otherwise normalizing the channels independently may not be a good choice.
            .filter('normalize', independence=1.0)
            #.filter('reverse')
            # YUV444p is the alternative to rgb24, but the pretrained network expects rgb images.
            #.output('pipe:', format='rawvideo', pix_fmt='yuv444p')
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
            .run_async(pipe_stdout=True, quiet=True)
        )
        # Generator loop
        # TODO FIXME Use the sampling options
        # The first frame will be frame number 1
        frame = 0
        # Need to read in all frames.
        while True:
            for target_idx, target_frame in enumerate(target_samples):
                # Get ready to fetch the next frame
                partial_sample = []
                sample_frames = []
                # Use the same crop location for each sample in multiframe sequences.
                crop_x = random.choice(range(0, 2 * self.crop_noise + 1))
                crop_y = random.choice(range(0, 2 * self.crop_noise + 1))
                while len(partial_sample) < self.frames_per_sample:
                    in_bytes = process1.stdout.read(in_width * in_height * self.channels)
                    if in_bytes:
                        # Numpy frame conversion either happens during background subtraction or
                        # later during sampling
                        np_frame = None
                        # Apply background subtraction if requested
                        if self.bg_subtractor is not None:
                            # Convert to numpy
                            np_frame = numpy.frombuffer(in_bytes, numpy.uint8)
                            fgMask = self.bg_subtractor.apply(np_frame)
                            # Curious use of a bitwise and involving the image and itself. Could use
                            # a masked select instead.
                            masked = bitwise_and(np_frame, np_frame, mask=fgMask)
                            np_frame = masked.clip(max=255).astype(numpy.uint8)

                        # Check if this frame will be sampled.
                        # The sample number is from the list of available samples, starting from 0. Add the
                        # begin frame to get the actual desired frame number.
                        # Making some variables here for clarity below
                        sample_in_progress = 0 < len(partial_sample)
                        if ((frame == target_frame or
                            (sample_in_progress and (frame - target_frame) %
                            (self.frame_interval + 1) == 0))):
                            # Convert to numpy, and then to torch.
                            if np_frame is None:
                                np_frame = numpy.frombuffer(in_bytes, numpy.uint8)
                            in_frame = torch.tensor(data=np_frame, dtype=torch.uint8,
                                ).reshape([1, in_height, in_width, self.channels])
                            # Apply the random crop
                            in_frame = in_frame[:, crop_y:crop_y+self.out_height, crop_x:crop_x+self.out_width, :]
                            in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)
                            partial_sample.append(in_frame)
                            sample_frames.append(str(frame))
                        frame += 1
                    else:
                        # Somehow we reached the end of the video without collected all of the samples.
                        print(f"Warning: reached the end of the video but only collected {target_idx}/{self.num_samples} samples")
                        print(f"Warning: ended during sample beginning with frame {target_frame} on frame {frame}")
                        process1.wait()
                        return
                # If multiple frames are being returned then concat them along the channel
                # dimension. Otherwise just return the single frame.
                if 1 == self.frames_per_sample:
                    yield partial_sample[0], self.path, sample_frames
                else:
                    yield torch.cat(partial_sample), self.path, sample_frames
            print(f"Collected {target_idx + 1} frames.")
            print(f"The final frame was {frame}")
            # Read any remaining samples
            while in_bytes:
                in_bytes = process1.stdout.read(self.out_width * self.out_height * self.channels)
            process1.wait()
            return

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
            sampler = VideoSampler(
                video_path=path, num_samples=args.samples, frames_per_sample=args.frames_per_sample,
                frame_interval=args.interval, out_width=args.width, out_height=args.height,
                crop_noise=args.crop_noise, scale=args.scale, crop_x_offset=args.crop_x_offset,
                crop_y_offset=args.crop_y_offset, channels=args.out_channels,
                begin_frame=row[beginf_col], end_frame=row[endf_col],
                bg_subtract=args.background_subtraction)
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
                        "metadata.txt": metadata.encode('utf-8')
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
                        "metadata.txt": metadata.encode('utf-8')
                    }
                    for i in range(args.frames_per_sample):
                        sample[f"{i}.png"] = buffers[i].getbuffer()

                datawriter.write(sample)

datawriter.close()
