import argparse

"""
ArgParser.py

This module is responsible for parsing command-line arguments for the pipeline script. It uses the argparse library to define and handle various parameters required for different stages of the pipeline.

Functions:
    - get_args: Parses and returns the command-line arguments.

Command-line Arguments:
    - --data_path: Path to the data, default is ".".
    - --start: Start the pipeline at the given step, default is 0.
    - --end: End the pipeline at the given step, default is 6.
    - --background-subtraction-type: Background subtraction type to use, choices are "MOG2" or "KNN", default is None.
    - --width: Width of the images, default is 960.
    - --height: Height of the images, default is 720.
    - --number-of-samples: The number of samples max that will be gathered by the sampler, default is 40000.
    - --max-workers-video-sampling: The number of workers to use for the multiprocessing of the sampler, default is 10.
    - --frames-per-sample: The number of frames per sample, default is 1.
    - --normalize: Normalize the images, default is True.
    - --out-channels: The number of output channels, default is 1.
    - --k: Number of folds for cross-validation, default is 3.
    - --model: Model to use, default is "alexnet".
    - --fps: Frames per second, default is 25.
    - --starting-frame: Starting frame, default is 1.
    - --frame-interval: Space between frames, default is 0.
    - --seed: Seed to use for randomizing the data sets, default is "01011970".
    - --only_split: Set to finish after splitting the CSV, default is False.
    - --crop_x_offset: The offset (in pixels) of the crop location on the original image in the x dimension, default is 0.
    - --crop_y_offset: The offset (in pixels) of the crop location on the original image in the y dimension, default is 0.
    - --training_only: Only generate the training set files, default is False.
    - --files: Name of the log files that one wants to use, default is None.
    - --max-workers-frame-counter: The number of workers to use for the multiprocessing of the frame counter, default is 20.
    - --max-workers-background-subtraction: The number of workers to use for the multiprocessing of the background subtraction, default is 10.

Usage:
    This script is intended to be used as part of a larger pipeline. It is typically invoked from the command line or another script, such as master_run.py.

Example:
    python ArgParser.py --data_path /path/to/data --start 0 --end 6 --width 960 --height 720

"""


def get_args():
    description = """
    Runs the pipeline that runs the model on the data.
    \n
    This programs expects the log files to be named of the form logNo.txt, logPos.txt, logNeg.txt.
    \n
    This script automatically converts the videos to .mp4, and then runs the pipeline on the data, type of video can either be mp4 or h264.
    \n
    This program also expects that you are running this on the ilab servers, with the anaconda environment of
    /koko/system/anaconda/envs/python38/bin:$PATH and /koko/system/anaconda/envs/python39/bin:$PATH.
    \n\n\n\n\n\n\n\n
    """
    poem = """    One file to rule them all,
    one file to find them,
    One file to bring them all,
    and in the data directory train them;
    In the Land of ilab where the shadows lie."""

    # truncating the log file

    parser = argparse.ArgumentParser(description=description, epilog=poem)
    parser.add_argument(
        "--data_path",
        type=str,
        help='Path to the data, default "."',
        default=".",
        required=False,
    )
    parser.add_argument(
        "--start",
        type=int,
        help="(unifier)Start the pipeline at the given step, default 0",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="(unifier)end the pipeline at the given step, default 6 (will not stop)",
        default=6,
        required=False,
    )
    # for background subtraction
    parser.add_argument(
        "--background-subtraction-type",
        choices=["MOG2", "KNN"],
        required=False,
        default=None,
        type=str,
        help="(background subtraction)Background subtraction type to use, default None, you can either choose MOG2 or KNN",
    )
    # for make_validation_training
    parser.add_argument(
        "--width",
        type=int,
        help="(splitting the data) Width of the images, default 960",
        default=960,
        required=False,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="(splitting the data) Height of the images, default 720",
        default=720,
        required=False,
    )

    # for sampling
    parser.add_argument(
        "--number-of-samples",
        type=int,
        help="(sampling)the number of samples max that will be gathered by the sampler, defalt=40000",
        default=40000,
    )
    parser.add_argument(
        "--max-workers-video-sampling",
        type=int,
        help="(sampling)The number of workers to use for the multiprocessing of the sampler, default=15",
        default=7,
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        help="(sampling, splitting the data)The number of frames per sample, default=1",
        default=1,
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        help="(sampling) normalize the images, default=True",
        default=True,
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        help="(sampling) The number of output channels, default=1",
        default=1,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="(making the splits) Number of folds for cross validation, default 3",
        default=3,
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="(making the splits) model to use, default alexnet",
        default="alexnet",
        required=False,
    )

    # CREATING THE DATASET
    parser.add_argument(
        "--fps",
        type=int,
        help="(dataset creation) frames per second, default 25",
        default=25,
    )
    parser.add_argument(
        "--starting-frame",
        type=int,
        help="(dataset creation) starting frame, default 1",
        default=1,
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        help="(dataset creation) space between frames, default 0",
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=str,
        default="01011970",
        help="(making the splits) Seed to use for randominizing the data sets, default: 01011970",
    )
    parser.add_argument(
        "--only_split",
        required=False,
        default=False,
        action="store_true",
        help="(making the splits) Set to finish after splitting the csv, default: False",
    )
    parser.add_argument(
        "--crop_x_offset",
        type=int,
        required=False,
        default=0,
        help="(making the splits) The offset (in pixels) of the crop location on the original image in the x dimension, default 0",
    )
    parser.add_argument(
        "--crop_y_offset",
        type=int,
        required=False,
        default=0,
        help="(making the splits) The offset (in pixels) of the crop location on the original image in the y dimension, default 0",
    )
    parser.add_argument(
        "--training_only",
        type=bool,
        required=False,
        default=False,
        help="(making the splits) only generate the training set files, default: False",
    )
    parser.add_argument(
        "--files",
        type=str,
        help="(dataset creation) name of the log files that one wants to use, default logNo.txt, logNeg.txt, logPos.txt",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--max-workers-frame-counter",
        type=int,
        help="(frame counting) The number of workers to use for the multiprocessing of the frame counter, default=20",
        default=20,
        required=False,
    )

    parser.add_argument(
        "--max-workers-background-subtraction",
        type=int,
        help="(background subtraction) The number of workers to use for the multiprocessing of the background subtraction, default=10",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="(training) The number of epochs to train the model, default=10",
        default=10,
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="(sampling) Crop the images to the correct size",
        default=False,
    )
    parser.add_argument(
        "--y-offset",
        type=int,
        help="The y offset for the crop, default=0",
        default=0,
    )
    parser.add_argument(
        "--out-width",
        type=int,
        help="The width of the output image, default=400",
        default=400,
    )
    parser.add_argument(
        "--out-height",
        type=int,
        help="The height of the output image, default=400",
        default=400,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information, activates debug for logger (and other scripts), defgault=False",
        default=False,
    )

    args = parser.parse_args()
    return args
