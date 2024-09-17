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

    assert provider.imageSize() == (720, 1280)

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

    assert provider.imageSize() == (720, 1280)

    # Try to get all of the frames through iteration
    found_frames = 0
    for frame in provider:
        found_frames += 1
        assert frame.ndim == 3
        assert frame.shape == (720, 1280, 1)
    assert found_frames == desired_frames

