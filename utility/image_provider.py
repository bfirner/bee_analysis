"""
Data provider for images and videos.

Copyright © 2024 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.
Despite the name, it may be useful for general machine learning.
"""

import av
import numpy
import os

from PIL import Image


def getImageProvider(data_string, **kwargs):
    """Return a video or image loader based upon the extension."""
    extension = os.path.splitext(data_string)[1]
    if extension in ['.gif', '.jpg', '.jpeg', '.png', '.tiff', '.webm']:
        return ImageReader(data_string, **kwargs)
    elif extension in ['.avi', '.h264', '.mjpeg', '.mkv', '.mpeg', '.mp4']:
        return VideoReader(data_string, **kwargs)
    else:
        raise RuntimeError("Unrecognized data extension: {}".format(extension))


class ImageReader:
    """Load a dataset of individual images with frame numbers in the file names."""

    def __init__(self, path, first_frame = 0, target_format='RGB'):
        """
        Arguments:
            data_string (str): A format string ready for frame number substitution. E.g. 'dir/image_{:05d}.png'
        """
        self.path = path
        self.first_frame = first_frame
        self.cur_idx = first_frame

        if target_format in ['bgr', 'rgb']:
            target_format = target_format.upper()
        elif target_format == 'gray':
            target_format = 'L'
        self.format = target_format

        # Set the total frames
        check_idx = first_frame
        while self.hasFrame(check_idx):
            check_idx += 1
        self.total_frames = check_idx - first_frame

        test_frame = self.getFrame(self.first_frame)
        self.height = test_frame.shape[0]
        self.width = test_frame.shape[1]

    def hasFrame(self, idx):
        return os.path.exists(self.path.format(idx))

    def getFrame(self, idx):
        if self.hasFrame(idx):
            img_array = numpy.array(Image.open(self.path.format(idx)).convert(self.format)).astype(numpy.float32) / 255.0
            # If there is only 1 channel then it gets dropped, so add it back.
            if 2 == img_array.ndim:
                return numpy.expand_dims(img_array, 2)
            else:
                return img_array
        else:
            raise RuntimeError("Requested frame ({}) does not exist.".format(idx))

    def totalFrames(self):
        return self.total_frames

    def imageSize(self):
        """Returns the image height and width as a tuple."""
        return self.width, self.height

    def __len__(self):
        return self.totalFrames()

    def __iter__(self):
        for framenum in range(self.first_frame, self.first_frame + self.totalFrames()):
            yield self.getFrame(framenum)


class VideoReader:
    """Load a dataset of images from a video."""

    def __init__(self, path, target_format='rgb24'):
        """
        Arguments:
            path (str): Path to the video
            target_format (str): Target video format. Should be one of the formats from `ffmpeg -pix_fmts`
        """
        self.path = path

        # To see all possible formats, run `ffmpeg -pix_fmts`
        if target_format in ['bgr', 'rgb']:
            target_format = target_format + "24"
        self.format = target_format

        # PyAV raises its own error upon failure, no need to check.
        self.container = av.open(path)
        self.width = self.container.streams.video[0].width
        self.height = self.container.streams.video[0].height
        # Loop through once to count the number of frames
        self.total_frames = 0
        for frame in self.container.decode(video=0):
            self.total_frames += 1
        self.container.close()

        self.container = None
        self.cur_frame = 0

    def hasFrame(self, idx):
        return idx >= 0 and idx < self.total_frames

    def getFrame(self, idx):
        if self.container is None:
            self.container = av.open(self.path)
            self.cur_frame = 0
        # Support seeking, as described here: https://pyav.org/docs/stable/api/container.html
        # and here: https://github.com/PyAV-Org/PyAV/discussions/1113
        if self.cur_frame + 1 != idx:
            # The seek function wants the presentation timestamp (or PTS)
            # All times must be in the video time base
            time_base = self.container.streams.video[0].time_base
            framerate = self.container.streams.video[0].average_rate
            desired_frame_sec = idx/framerate
            video_ts = round(desired_frame_sec / time_base)
            try:
                # Check the frame that container.seek went to and see if it is the right frame or merely the nearest key frame.
                self.container.seek(offset=video_ts, stream=self.container.streams.video[0])
                frame = next(self.container.decode(video=0))
                returned_frame = int(frame.pts * time_base * framerate)
            except av.error.PermissionError:
                print("Seeking not possible, reopening file.")
                # Can't seek with this file type, need to go back to the beginning
                self.container.close()
                self.container = av.open(self.path)
                time_base = self.container.streams.video[0].time_base
                framerate = self.container.streams.video[0].average_rate
                frame = next(self.container.decode(video=0))
                self.cur_frame = 0
                returned_frame = 0
             

            # Seek from the keyframe to the desired frame if this didn't match
            for _ in range(returned_frame, idx):
                frame = next(self.container.decode(video=0))
            self.cur_frame = idx
        else:
            self.cur_frame += 1
            frame = next(self.container.decode(video=0))
        return frame.to_ndarray(format=self.format).astype(numpy.float32) / 255.0

    def imageSize(self):
        """Returns the image height and width as a tuple."""
        return self.width, self.height

    def totalFrames(self):
        return self.total_frames

    def __len__(self):
        return self.totalFrames()

    def __iter__(self):
        iter_container = av.open(self.path)
        for frame in iter_container.decode(video=0):
            self.cur_frame += 1
            npframe = frame.to_ndarray(format=self.format).astype(numpy.float32) / 255.0
            del frame
            yield npframe

    def __del__(self):
        if self.container is not None:
            self.container.close()
