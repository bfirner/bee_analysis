#! /usr/bin/python3

"""
Utility functions and classes for video processing
"""

import ffmpeg
import math

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
