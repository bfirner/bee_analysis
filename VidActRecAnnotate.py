#! /usr/bin/python3

"""
This program will annotate given video data with predictions from a provided DNN. The area of the
image that is being fed to the neural network will be outlined and the prediction will be shown as
bar charts at the side of the image.
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

# for logging when where program was run
from subprocess import PIPE, run

# Helper function to convert to images
from torchvision import transforms
# For annotation drawing
from PIL import ImageDraw, ImageFont, ImageOps


from models.alexnet import AlexLikeNet
from models.bennet import BenNet
from models.resnet import (ResNet18, ResNet34)
from models.resnext import (ResNext18, ResNext34, ResNext50)
from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)

from utility.video_utility import (getVideoInfo, vidSamplingCommonCrop)

def commandOutput(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

parser = argparse.ArgumentParser(
    description="Annotate a given video.")
parser.add_argument(
    '--datalist',
    type=str,
    help=('A csv file with one or more video files and their labels.'
          ' Columns should be "file," "label," "begin," and "end".'
          ' The "begin" and "end" columns are in frames.'
          ' If the "begin" and "end" columns are not present,'
          ' then the label will be applied to the entire video.'))
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
    '--height, with --crop_x_offset and --crop_y_offset for more options.')
parser.add_argument(
    '--scale',
    type=float,
    required=False,
    default=1.0,
    help='Scaling to apply to each dimension (before cropping). A value of 0.5 will yield 0.25 resolution.')
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
    help='Frames to skip between processed and recorded frames.')
parser.add_argument(
    '--frames_per_sample',
    type=int,
    required=False,
    default=1,
    help='Number of frames in each sample.')
parser.add_argument(
    '--dnn_channels',
    type=int,
    required=False,
    choices=[1, 3],
    default=1,
    help='Channels feed to the DNN.')
parser.add_argument(
    '--label_classes',
    type=int,
    required=False,
    default=3,
    help='Number of label classes predicted by the model.')
parser.add_argument(
    '--class_names',
    type=str,
    nargs='+',
    required=False,
    help='Label class names.')
parser.add_argument(
    '--normalize',
    required=False,
    default=False,
    action="store_true",
    help=("Normalize inputs: input = (input - mean) / stddev. "
        "Note that VidActRecDataprep is already normalizing so this may not be required."))
parser.add_argument(
    '--resume_from',
    type=str,
    required=True,
    help='Model weights to restore.')
parser.add_argument(
    '--modeltype',
    type=str,
    required=False,
    default="resnext18",
    choices=["alexnet", "resnet18", "resnet34", "bennet", "resnext50", "resnext34", "resnext18",
    "convnextxt", "convnextt", "convnexts", "convnextb"],
    help="Model to use for training.")
parser.add_argument(
    '--loss_fun',
    required=False,
    default='CrossEntropyLoss',
    choices=['NLLLoss', 'BCEWithLogitsLoss', 'CrossEntropyLoss', 'L1Loss', 'MSELoss', 'BCELoss'],
    type=str,
    help="Loss function to use during training.")
parser.add_argument(
    '--label_offset',
    required=False,
    default=0,
    type=int,
    help='The starting value of classes when training with cls labels (the labels value is "cls").')
parser.add_argument(
    '--background_subtraction',
    type=str,
    required=False,
    choices=['none', 'mog2', 'knn'],
    default='none',
    help='Background subtraction algorithm to apply to the input video, or none.')

args = parser.parse_args()

# Network outputs may need to be postprocessed for evaluation if some postprocessing is being done
# automatically by the loss function.
if 'CrossEntropyLoss' == args.loss_fun:
    nn_postprocess = torch.nn.Softmax(dim=1)
elif 'BCEWithLogitsLoss' == args.loss_fun:
    nn_postprocess = torch.nn.Sigmoid()
else:
    # Otherwise just use an identify function.
    nn_postprocess = lambda x: x

# these next clauses are for scripts so that we have a log of what was called on what machine and when.
python_log =  commandOutput("which python3")
machine_log = commandOutput("uname -a")
date_log = commandOutput("date")

print("Log: Program_args: ",end='')
for theArg in sys.argv :
    print(theArg + " ",end='')
print(" ")

print("Log: Started: ",date_log)
print("Log: cwd: ", os.getcwd() )
print("Log: Machine: ",machine_log)
print("Log: Python_version: ",python_log)

class VideoAnnotator:

    def __init__(self, video_labels, net, frame_interval, frames_per_sample, out_width=None,
            out_height=None, scale=1.0, crop_x_offset=0, crop_y_offset=0, channels=3,
             begin_frame=None, end_frame=None, output_name="annotated.mp4", bg_subtract="none"):
        """
        Samples have no overlaps. For example, a 10 second video at 30fps has 300 samples of 1
        frame, 150 samples of 2 frames with a frame interval of 0, or 100 samples of 2 frames with a
        frame interval of 1.
        Arguments:
            video_labels  (VideoLabels): VideoLabels with video path and frame class labels.
            net   (torch.nn.Module): The neural network used for classification and annotation.
            frame_interval    (int): Number of frames to skip between each sampled frame.
            frames_per_sample (int):  Number of frames in each sample.
            out_width     (int): Width of output images (for the DNN), or the original width if None.
            out_height    (int): Height of output images (for the DNN), or the original height if None.
            scale       (float): Scale factor of each dimension
            crop_x_offset (int): x offset of crop, in pixels, from the original image
            crop_y_offset (int): y offset of crop, in pixels, from the original image
            channels      (int): Numbers of channels (3 for RGB or 1 luminance/Y/grayscale/whatever)
            begin_frame   (int): First frame to sample.
            end_frame     (int): Final frame to sample.
            output_name   (str): Name for the output annotated video.
            bg_subtract   (str): Type of background subtraction to use (mog2 or knn), or none.
        """
        # TODO This should be reusing components of the VideoSampler class from VidActRecDataprep.py
        self.path = video_path
        self.video_labels = video_labels
        self.frames_per_sample = frames_per_sample
        self.frame_interval = frame_interval
        self.channels = channels
        self.scale = scale
        self.output_name = output_name
        self.normalize = False
        print(f"Processing {video_path}")
        # Probe the video to find out some metainformation
        self.width, self.height, self.total_frames = getVideoInfo(video_path)

        self.out_width, self.out_height, self.crop_x, self.crop_y = vidSamplingCommonCrop(
            self.height, self.width, out_height, out_width, self.scale, crop_x_offset, crop_y_offset)

        # Background subtraction will require openCV if requested.
        self.bg_subtractor = None
        if ('none' != bg_subtract):
            from cv2 import (createBackgroundSubtractorMOG2,
                             createBackgroundSubtractorKNN)
            if 'mog2' == bg_subtract:
                self.bg_subtractor = createBackgroundSubtractorMOG2()
            elif 'knn' == bg_subtract:
                self.bg_subtractor = createBackgroundSubtractorKNN()

        if begin_frame is None:
            self.begin_frame = 1
        else:
            self.begin_frame = int(begin_frame)

        if end_frame is None:
            self.end_frame = self.total_frames
        else:
            # Don't attempt to sample more frames than there exist.
            self.end_frame = min(int(end_frame), self.total_frames)
        print(f"Video begin and end frames are {self.begin_frame} and {self.end_frame}")


    def setSeed(self, seed):
        """Set the seed used for sample generation in the iterator."""
        self.seed = seed


    def process_video(self):
        """An iterator that yields frames.

        Decode the video and write the annotated result.
        """
        # Open the video
        # It is a bit unfortunate the we decode what is probably a YUV stream into rgb24, but this
        # is what PIL supports easily. It is only really detrimental when we want just the Y
        # channel.
        if 3 == self.channels:
            pix_fmt='rgb24'
        else:
            pix_fmt='gray'
        in_width = self.out_width
        in_height = self.out_height

        # Prepare a font for annotations. Just using the default font for now.
        try:
            font = ImageFont.truetype(font="DejaVuSans.ttf", size=14)
        except OSError:
            font = ImageFont.load_default()

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

        # Begin the video input process from ffmpeg.
        input_process = (
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
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
            .run_async(pipe_stdout=True, quiet=True)
        )
        # Begin an output process from ffmpeg.
        info_width = in_width // 3
        output_process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{in_width+info_width}x{in_height}')
            .output(self.output_name, pix_fmt='rgb24')
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )
        # Generator loop
        # The first frame will be frame number 1
        frame = 0
        # Discard early frames
        while frame < self.begin_frame:
            in_bytes = input_process.stdout.read(in_width * in_height * self.channels)
            if in_bytes:
                frame += 1
        # Read in all frames that should be processed
        sample_frames = []
        while frame < self.end_frame:
            # Fetch the next frame sample that can be sent through the neural network
            in_bytes = input_process.stdout.read(in_width * in_height * self.channels)
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

                frame += 1
                # Convert to numpy (if not already handled by the background subtractor), and then to torch.
                if np_frame is None:
                    np_frame = numpy.frombuffer(in_bytes, numpy.uint8)
                in_frame = torch.tensor(data=np_frame, dtype=torch.uint8,
                    ).reshape([1, in_height, in_width, self.channels])
                in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float).cuda()
                sample_frames.append(in_frame)


            else:
                # We reached the end of the video before reaching the desired end frame somehow.
                input_process.wait()
                # Close the output and return.
                output_process.stdin.close()
                output_process.wait()
                return

            if len(sample_frames) == self.frames_per_sample:
                # If multiple frames are being used for inference then concat them along the channel
                # dimension. Otherwise just use the single frame.

                if 1 == self.frames_per_sample:
                    image_input = sample_frames[0]
                else:
                    # Concatenate along the channel dimension since the first dimension will be
                    # treated as the batch size.
                    image_input = torch.cat(sample_frames, 1)

                # Get the label for this frame. Multiframe inputs have the label of the newest frame.
                label = self.video_labels.getLabel(frame)

                # Now normalize the image and send it through the DNN. Then annotate the image with
                # the label and result.
                # Visualization masks are not supported with all model types yet.
                with torch.no_grad():
                    if args.normalize:
                        # Normalize per channel, so compute over height and width
                        v, m = torch.var_mean(image_input, dim=(image_input.dim()-2, image_input.dim()-1), keepdim=True)
                        net_input = (image_input - m) / v
                    else:
                        net_input = image_input
                    out, mask = net.vis_forward(net_input)
                    out = nn_postprocess(out)

                # Reconvert to an image for the output video stream
                display_frame = sample_frames[-1][0]
                # Convert to a color image is necessary
                if 3 != self.channels:
                    display_frame = display_frame.repeat(3, 1, 1)

                # Draw bounding boxes around every large group of features
                # Add all of the pixel features greater than 1% of the total into a set
                mask_total = mask[0].sum().item()
                #mask_captures = mask[0,0] > (mask_total/100.0)
                mask_captures = mask[0,0] > 0.70
                # Turn the image tensor green at the masked locations.
                display_frame[1].masked_fill_(mask_captures, 255.0)

                cur_image = transforms.ToPILImage()(display_frame.cpu()/255.0).convert('RGB')

                # Segment mask captures into bounding boxes.
                #mask_pixels = [(i, j) for i in range(mask.size(2)) for j
                #        in range(mask.size(3)) if mask_captures[i,j]]
                # Do bfs or dfs to cluster them
                # Draw bounding boxes around the clusters

                # Part 2: Assign clusters to classes.
                # Before adding features into a set, create a set for each class.
                # Go through the net.classifier part of the DNN (the linear layers) to assign
                # classes by backpropping through each class prediction.

                # Pad an empty space to the right.
                padded_image = ImageOps.pad(cur_image, (in_width + info_width, in_height), centering=(0,0))

                # Get the drawing context
                cont = ImageDraw.Draw(padded_image)
                # Annotate with the label
                rows = len(self.video_labels.class_names) + 1
                cont.text(((in_width + info_width//2), in_height//rows), f"Label: {label}",
                        fill=(235, 235, 235), font=font, anchor="mm")

                for row in range(2, rows):
                    lname = self.video_labels.class_names[row-1]
                    lscore = out[0,row-2].item()
                    cont.text(((in_width + info_width//2), row * in_height//rows),
                        f"{lname} score: {round(lscore, 3)}", fill=(235, 235, 235), font=font, anchor="mm")

                # Write the frame
                output_process.stdin.write(
                    padded_image
                    .tobytes()
                )

                # Remove the consumed frame from the samples.
                sample_frames = sample_frames[1:]

        # Read any remaining samples to finish the video decoding process.
        while in_bytes:
            in_bytes = input_process.stdout.read(self.out_width * self.out_height * self.channels)
        input_process.wait()
        output_process.stdin.close()
        output_process.wait()


image_size = (args.dnn_channels * args.frames_per_sample, args.height, args.width)

# Model setup stuff
if 'alexnet' == args.modeltype:
    net = AlexLikeNet(in_dimensions=image_size, out_classes=args.label_classes, linear_size=512).cuda()
elif 'resnet18' == args.modeltype:
    net = ResNet18(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True).cuda()
elif 'resnet34' == args.modeltype:
    net = ResNet34(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True).cuda()
elif 'bennet' == args.modeltype:
    net = BenNet(in_dimensions=image_size, out_classes=args.label_classes).cuda()
elif 'resnext50' == args.modeltype:
    net = ResNext50(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True).cuda()
elif 'resnext34' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext34(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=False,
            use_dropout=False).cuda()
elif 'resnext18' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext18(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True,
            use_dropout=False).cuda()
elif 'convnextxt' == args.modeltype:
    net = ConvNextExtraTiny(in_dimensions=image_size, out_classes=args.label_classes).cuda()
elif 'convnextt' == args.modeltype:
    net = ConvNextTiny(in_dimensions=image_size, out_classes=args.label_classes).cuda()
elif 'convnexts' == args.modeltype:
    net = ConvNextSmall(in_dimensions=image_size, out_classes=args.label_classes).cuda()
elif 'convnextb' == args.modeltype:
    net = ConvNextBase(in_dimensions=image_size, out_classes=args.label_classes).cuda()
print(f"Model is {net}")

# See if the model weights can be restored.
if args.resume_from is not None:
    checkpoint = torch.load(args.resume_from)
    # Remove vis_layers from the checkpoint to support older models with the current code.
    vis_names = [key for key in list(checkpoint['model_dict'].keys()) if key.startswith("vis_layers")]
    for key in vis_names:
        del checkpoint['model_dict'][key]
    missing_keys, unexpected_keys = net.load_state_dict(checkpoint["model_dict"], strict=False)
    if (unexpected_keys):
        raise RuntimeError(f"Found unexpected keys in model checkpoint: {unexpected_keys}")
    # Update the weights for the vis mask layers
    net.createVisMaskLayers(net.output_sizes)
    net = net.cuda()

# Always use the network in evaluation mode.
net.eval()

class LabelRange:
    """LabelRange class used to store all of the label data for a video."""
    def __init__(self, labelclass, beginframe, endframe):
        self.labelclass = labelclass
        self.beginframe = beginframe
        self.endframe = endframe

    def __lt__(self, other):
        return self.beginframe < other.beginframe

class VideoLabels:
    """A video and all of their labels."""
    def __init__(self, videoname, class_names):
        self.videoname = videoname
        self.class_names = class_names
        self.labels = []

    def addLabel(self, labelrange):
        """Add label range information for this video."""
        self.labels.append(labelrange)
        list.sort(self.labels)

    def getLabel(self, frame):
        """Get the label for a frame number. Frame numbers begin at 1."""
        # Return "none" if there is no class information for the given frame
        if (0 == len(self.labels) or frame < self.labels[0].beginframe or
                frame > self.labels[-1].endframe):
            return "none"
        # Advance the index until we encounter a label range that should have the label for this
        # frame.
        idx = 0
        while (self.labels[idx].endframe < frame):
            idx += 1
        # It's possible that there is a gap between labels.
        if (self.labels[idx].beginframe > frame):
            return "none"
        # Otherwise we have finally found the label for this frame.
        return self.class_names[self.labels[idx].labelclass]

# Create strings with label class names if none were provided.
if not args.class_names:
    args.class_names = []
# Index 0 does not correspond to a valid class label
args.class_names = (["none"] * args.label_offset) + args.class_names

# Make sure that each label value has a string
for i in range(args.label_classes):
    # There should be one more class name than labels because we inserted a "none" class at the 0
    # index position.
    if len(args.class_names)-1 <= i:
        # If this is label 0 then name it 'class 1', and so on
        args.class_names.append(f"class {i+1}")

with open(args.datalist, newline='') as datacsv:
    conf_reader = csv.reader(datacsv)
    header = next(conf_reader)
    # Remove all spaces from the header strings
    header = [''.join(col.split(' ')) for col in header]
    file_col = header.index('file')
    class_col = header.index('class')
    beginf_col = header.index('beginframe')
    endf_col = header.index('endframe')
    # TODO Loop through and group all entries that come from the same video.
    # Each video will have its own set of label ranges
    video_labels = {}

    for row in conf_reader:
        # Read the next video
        # Make sure that this line is sane
        if 4 != len(row):
            print(f"Row '{row}' does not have the correct number of columns (4).")
        else:
            path = row[file_col]
            if (path not in video_labels):
                video_labels[path] = VideoLabels(path, args.class_names)
            video_labels[path].addLabel(LabelRange(int(row[class_col]), int(row[beginf_col]), int(row[endf_col])))

    # Now play each video and annotate them with results
    for video_path, video_label in video_labels.items():
        begin_frame = video_label.labels[0].beginframe
        end_frame = video_label.labels[-1].endframe
        annotated_name = "annotated_" + os.path.basename(video_path)
        sampler = VideoAnnotator(
            video_labels=video_label, net=net, frame_interval=args.interval,
            frames_per_sample=args.frames_per_sample, out_width=args.width, out_height=args.height,
            scale=args.scale, crop_x_offset=args.crop_x_offset, crop_y_offset=args.crop_y_offset,
            channels=args.dnn_channels, begin_frame=begin_frame, end_frame=end_frame,
            output_name=annotated_name)

        sampler.process_video()
