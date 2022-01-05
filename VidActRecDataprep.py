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
    help='Width of output images.')
parser.add_argument(
    '--height',
    type=int,
    required=False,
    default=224,
    help='Height of output images.')
parser.add_argument(
    '--resize-strategy',
    type=str,
    required=False,
    default='crop',
    choices=['crop', 'scale'],
    help='Strategy to match desired output size.')
parser.add_argument(
    '--crop_noise',
    type=int,
    required=False,
    default=0,
    help='The noise (in pixels) to randomly add to the crop location in both the x and y axis.')
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

args = parser.parse_args()

class VideoSampler:

    def __init__(self, video_path, num_samples, frames_per_sample, frame_interval,
            out_width=None, out_height=None, crop_noise=0, channels=3, begin_frame=None,
            end_frame=None):
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
            channels      (int): Numbers of channels (3 for RGB or 1 luminance/Y/grayscale/whatever)
            begin_frame   (int): First frame to possibly sample.
            end_frame     (int): Final frame to possibly sample.
        """
        self.path = video_path
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample
        self.frame_interval = frame_interval
        self.channels = channels
        print(f"Processing {video_path}")
        # Probe the video to find out some metainformation

        # Following advice from https://kkroening.github.io/ffmpeg-python/index.html
        # First find the size, then set up a stream.
        probe = ffmpeg.probe(self.path)['streams'][0]
        self.width = probe['width']
        self.height = probe['height']
        if out_width is not None:
            self.out_width = out_width
        else:
            self.out_width = self.width
        if out_height is not None:
            self.out_height = out_height
        else:
            self.out_height = self.height
        if out_width is None or out_height is None:
            self.crop_noise = 0
        else:
            self.crop_noise = crop_noise

        if 'duration' in probe:
            numer, denom = probe['avg_frame_rate'].split('/')
            self.frame_rate = float(numer) / float(denom)
            self.duration = float(probe['duration'])
            self.total_frames = math.floor(self.duration * self.frame_rate)
        else:
            # If the duration is not in the probe then we will need to read through the entire video
            # to get the number of frames.
            # It is possible that the "quiet" option to the python ffmpeg library may have a buffer
            # size problem as the output does not go to /dev/null to be discarded. The workaround
            # would be to manually poll the buffer.
            process1 = (
                ffmpeg
                .input(self.path)
                .output('pipe:', format='rawvideo', pix_fmt='gray')
                #.output('pipe:', format='rawvideo', pix_fmt='yuv420p')
                .run_async(pipe_stdout=True, quiet=True)
            )
            # Count frames
            frame = 0
            while True:
                # Using pix_fmt='gray' we should get a single channel of 8 bits per pixel
                in_bytes = process1.stdout.read(self.width * self.height)
                if in_bytes:
                    frame += 1
                else:
                    process1.wait()
                    break
            self.total_frames = frame
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
        process1 = (
            ffmpeg
            .input(self.path)
            # The crop is automatically centered if the x and y parameters are not used.
            .filter('crop', out_w=in_width, out_h=in_height)
            # Full indepenence between color channels. The bee videos are basically a single color.
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
                        # Check if this frame will be sampled.
                        # The sample number is from the list of available samples, starting from 0. Add the
                        # begin frame to get the actual desired frame number.
                        # Making some variables here for clarity below
                        sample_in_progress = 0 < len(partial_sample)
                        if ((frame == target_frame or
                            (sample_in_progress and (frame - target_frame) %
                            (self.frame_interval + 1) == 0))):
                            # Convert to numpy, and then to torch.
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
        path = row[file_col]
        sampler = VideoSampler(
            video_path=path, num_samples=args.samples, frames_per_sample=args.frames_per_sample,
            frame_interval=args.interval, out_width=args.width, out_height=args.height,
            crop_noise=args.crop_noise, channels=args.out_channels,
            begin_frame=row[beginf_col], end_frame=row[endf_col])
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

            #img.save(f"frame_{sample_num}.png")
            # Need the input conditioner to normalize the channel means to 0.5 and stdev to 0.5
            #inputs = input_conditioner(images=cropped[0], return_tensors='pt')
            #inputs = {'pixel_values': cropped}
            #outputs = model(pixel_values=cropped)
            #print(f"Outputs logits max is {outputs.logits.argmax(-1).item()}")

datawriter.close()
# TODO FIXME add some random placement to the crop that should mitigate the risk of some temporal
# signals that may allow a DNN to cheat.
