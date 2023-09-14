#! /usr/bin/python3

"""
Utility functions and classes for video processing
"""

import ffmpeg
import math
import numpy
import random
import torch

def vidSamplingCommonCrop(height, width, out_height, out_width, scale, x_offset, y_offset):
    """
    Return the common cropping parameters used in dataprep and annotations.

    Arguments:
        height     (int): Height of the video
        width      (int): Width of the video
        out_height (int): Height of the output patch
        out_width  (int): Width of the output patch
        scale    (float): Scale applied to the original video
        x_offset   (int): x offset of the crop (after scaling)
        y_offset   (int): y offset of the crop (after scaling)
    Returns:
        out_width, out_height, crop_x, crop_y
    """
    
    if out_width is None:
        out_width = math.floor(width * scale)
    if out_height is None:
        out_height = math.floor(height * scale)

    crop_x = math.floor((width * scale - out_width)/2 + x_offset)
    crop_y = math.floor((height * scale - out_height)/2 + y_offset)

    return out_width, out_height, crop_x, crop_y


def getVideoInfo(video_path):
    """
    Get the total frames in a video.

    Arguments:
        video_path (str): The path to the video file.
    Returns:
        int: Width
        int: Height
        int: The total number of frames.
    """
    # Following advice from https://kkroening.github.io/ffmpeg-python/index.html
    # First find the size, then set up a stream.
    probe = ffmpeg.probe(video_path)['streams'][0]
    width = probe['width']
    height = probe['height']

    if 'duration' in probe:
        numer, denom = probe['avg_frame_rate'].split('/')
        frame_rate = float(numer) / float(denom)
        duration = float(probe['duration'])
        total_frames = math.floor(duration * frame_rate)
    else:
        # If the duration is not in the probe then we will need to read through the entire video
        # to get the number of frames.
        # It is possible that the "quiet" option to the python ffmpeg library may have a buffer
        # size problem as the output does not go to /dev/null to be discarded. The workaround
        # would be to manually poll the buffer.
        process1 = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='gray')
            #.output('pipe:', format='rawvideo', pix_fmt='yuv420p')
            .run_async(pipe_stdout=True, quiet=True)
        )
        # Count frames
        frame = 0
        while True:
            # Using pix_fmt='gray' we should get a single channel of 8 bits per pixel
            in_bytes = process1.stdout.read(width * height)
            if in_bytes:
                frame += 1
            else:
                process1.wait()
                break
        total_frames = frame
    return width, height, total_frames


class VideoSampler:

    def __init__(self, video_path, num_samples, frames_per_sample, frame_interval,
            out_width=None, out_height=None, crop_noise=0, scale=1.0, crop_x_offset=0,
             crop_y_offset=0, channels=3, begin_frame=None, end_frame=None,
             bg_subtract='none', normalize=True):
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
            normalize    (bool): True to normalize image channels (done independently)
        """
        self.path = video_path
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample
        self.frame_interval = frame_interval
        self.channels = channels
        self.scale = scale
        self.normalize = normalize

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
            )
            if self.normalize:
                # Full independence between color channels. The bee videos are basically a single color.
                # Otherwise normalizing the channels independently may not be a good choice.
                process1 = process1.filter('normalize', independence=1.0)

            process1 = (
                process1
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

        # Reading videos slows down as more frames have been read, so read them in chunks.
        frame_batch_size = 2000
        cur_end_frame = min(frame_batch_size, self.end_frame)
        process1 = (
            ffmpeg
            .input(self.path)
            # Read the next chunk
            .trim(start_frame=1, end_frame=cur_end_frame)
            # Scale
            .filter('scale', self.scale*self.width, -1)
            # The crop is automatically centered if the x and y parameters are not used.
            .filter('crop', out_w=in_width, out_h=in_height, x=self.crop_x, y=self.crop_y)
            # Full independence between color channels. The bee videos are basically a single color.
            # Otherwise normalizing the channels independently may not be a good choice.
        )
        if self.normalize:
            # Full independence between color channels. The bee videos are basically a single color.
            # Otherwise normalizing the channels independently may not be a good choice.
            process1 = process1.filter('normalize', independence=1.0)
        process1 = (
            process1
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
        in_bytes = True
        while in_bytes:
            for target_idx, target_frame in enumerate(target_samples):
                # Get ready to fetch the next frame
                partial_sample = []
                sample_frames = []
                # Use the same crop location for each sample in multiframe sequences.
                crop_x = random.choice(range(0, 2 * self.crop_noise + 1))
                crop_y = random.choice(range(0, 2 * self.crop_noise + 1))
                while len(partial_sample) < self.frames_per_sample and frame < self.end_frame:
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
                    elif (cur_end_frame < self.end_frame):
                        # Let the previous process end.
                        process1.wait()
                        # Go to the next chunk
                        cur_end_frame = min(frame_batch_size, self.end_frame)
                        next_end_frame = min(cur_end_frame+frame_batch_size, self.end_frame)
                        process1 = (
                            ffmpeg
                            .input(self.path)
                            # Read the next chunk
                            .trim(start_frame=cur_end_frame+1, end_frame=next_end_frame)
                            # Scale
                            .filter('scale', self.scale*self.width, -1)
                            # The crop is automatically centered if the x and y parameters are not used.
                            .filter('crop', out_w=in_width, out_h=in_height, x=self.crop_x, y=self.crop_y)
                            # Full independence between color channels. The bee videos are basically a single color.
                            # Otherwise normalizing the channels independently may not be a good choice.
                        )
                        if self.normalize:
                            # Full independence between color channels. The bee videos are basically a single color.
                            # Otherwise normalizing the channels independently may not be a good choice.
                            process1 = process1.filter('normalize', independence=1.0)
                        process1 = (
                            process1
                            # YUV444p is the alternative to rgb24, but the pretrained network expects rgb images.
                            #.output('pipe:', format='rawvideo', pix_fmt='yuv444p')
                            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                            .run_async(pipe_stdout=True, quiet=True)
                        )
                        cur_end_frame = next_end_frame
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
