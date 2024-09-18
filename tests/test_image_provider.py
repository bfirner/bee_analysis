import numpy
import os
import pytest
import shlex
import subprocess

import utility.image_provider as ip

def testVideoReading(tmp_path):
    """Test that the video has the correct number of frames and that they can be read."""

    # Create the video in the tmp path
    desired_frames = 60
    subprocess.run(shlex.split("bash generate_test_video.sh {} {}".format(desired_frames, tmp_path)))

    video_path = os.path.join(tmp_path, "synthetic_test_video.mp4")
    provider = ip.getImageProvider(video_path)

    # Test number of frames
    assert provider.totalFrames() == desired_frames

    assert provider.imageSize() == (1280, 720)

    # Try to get all of the frames through iteration
    found_frames = 0
    for frame in provider:
        found_frames += 1
        assert frame.ndim == 3
        assert frame.shape == (720, 1280, 3)
    assert found_frames == desired_frames

def testImageReading(tmp_path):
    """Test that the video has the correct number of frames and that they can be read."""

    # Create the video in the tmp path
    desired_frames = 60
    subprocess.run(shlex.split("bash generate_test_video.sh {} {}".format(desired_frames, tmp_path)))

    video_path = os.path.join(tmp_path, "synthetic_test_{}.png")
    provider = ip.getImageProvider(video_path, first_frame=1, target_format='gray')

    # Test number of frames
    assert provider.totalFrames() == desired_frames

    assert provider.imageSize() == (1280, 720)

    # Try to get all of the frames through iteration
    found_frames = 0
    for frame in provider:
        found_frames += 1
        assert frame.ndim == 3
        assert frame.shape == (720, 1280, 1)
    assert found_frames == desired_frames

def testVideoSeeking(tmp_path):
    """Test that the video has the correct number of frames and that they can be read."""

    # Create the video in the tmp path
    desired_frames = 300
    subprocess.run(shlex.split("bash generate_test_video.sh {} {}".format(desired_frames, tmp_path)))

    video_path = os.path.join(tmp_path, "synthetic_test_video.mp4")
    provider = ip.getImageProvider(video_path)

    # Iterate through all frames
    frames = []
    for frame in provider:
        frames.append(frame)

    for test_idx in [0, 1, 0, 23, 1, 280, 180, 250]:
        numpy.testing.assert_array_equal(frames[test_idx], provider.getFrame(test_idx))

