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
import cv2
import numpy as np 

# Import efficient video reading tools
from utility.image_provider import VideoReader
from utility.patch_common import getCropCoords, imagePreprocessFromCoords
import gc  # for garbage collection
# for logging when where program was run
from subprocess import PIPE, run

# Helper function to convert to images
from torchvision import transforms
# For annotation drawing
from PIL import ImageDraw, ImageFont, ImageOps, Image


from models.alexnet import AlexLikeNet
from models.bennet import BenNet
from models.resnet import (ResNet18, ResNet34)
from models.resnext import (ResNext18, ResNext34, ResNext50)
from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)

from utility.model_utility import restoreModel
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


class EfficientVideoDecoder:
    """Efficient video decoder using VideoReader from image_provider."""

    def release(self):
        """Release resources."""
        # VideoReader doesn't have a release method, but we can close its container
        if hasattr(self.reader, 'container') and self.reader.container is not None:
            self.reader.container.close()
    
    def __init__(self, video_path, width, height, scale=1.0, crop_x=0, crop_y=0, 
                begin_frame=1, end_frame=None, frames_per_sample=5, planes=1, src='RGB'):
        # Store all parameters as instance attributes
        self.video_path = video_path
        self.width = width
        self.height = height
        self.scale = scale
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.begin_frame = begin_frame 
        self.frames_per_sample = frames_per_sample
        self.planes = planes  #Added planes and src as arguements
        self.src = src
        
        # Initialize the video reader
        self.reader = VideoReader(video_path)
        # Use direct properties instead of get_width/get_height methods
        video_width = self.reader.width
        video_height = self.reader.height
        
        # Set end frame if not specified - use totalFrames() instead of get_frame_count()
        if end_frame is None:
            self.end_frame = self.reader.totalFrames()
        else:
            self.end_frame = end_frame
            
        print(f"Video begin and end frames are {self.begin_frame} and {self.end_frame}")
        
        # Set up image processing parameters
        self.improc = {
            'size': (video_width, video_height),
            'scale': self.scale,
            'width': self.width,
            'height': self.height,
            'crop_x_offset': self.crop_x,
            'crop_y_offset': self.crop_y,
            'frames_per_sample': self.frames_per_sample, 
            'format': 'rgb24' 
        }
        
        # Calculate actual dimensions after scaling and cropping
        self.scale_w, self.scale_h, self.crop_coords = getCropCoords(self.improc)
        self.in_width = self.width
        self.in_height = self.height
        
        # Just set the current frame counter
        self.current_frame = self.begin_frame
        
    def read_frame(self):
        """Read a single frame from the video."""
        if self.current_frame >= self.end_frame:
            return None
        
        try:
            # Use getFrame instead of read
            frame = self.reader.getFrame(self.current_frame)
            self.current_frame += 1
            
            # Process frame (scale and crop)
            processed_frame = self.preprocess_frame(frame)
            return processed_frame
        except Exception as e:
            print(f"Failed to read frame at position {self.current_frame}: {e}")
            return None
        
    def preprocess_frame(self, frame):
        """Preprocess a frame using optimized OpenCV implementation."""
        # Use the optimized utility function
        processed_frame = imagePreprocessFromCoords(
            frame, 
            self.scale_w, 
            self.scale_h, 
            self.crop_coords,
            planes=self.planes,  # Set to 1 for grayscale (model expects 5 channels total)
            src=self.src  # VideoReader returns RGB format
        )
        
        # Convert to tensor with proper dimensions for the network
        tensor_frame = torch.tensor(
            data=processed_frame, 
            dtype=torch.float
        ).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        return tensor_frame

class VideoAnnotator:

    def __init__(self, video_labels, net, frame_interval, frames_per_sample, out_width=None,
        out_height=None, scale=1.0, crop_x_offset=0, crop_y_offset=0, channels=3,
         begin_frame=None, end_frame=None, output_name="annotated.mp4", bg_subtract="none"):
    
        # Store the neural network
        self.net = net
        
        # Use the correct path from the video_labels object
        video_path = video_labels.videoname
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

    
    def process_mask(self, mask, frame, in_width, in_height, display_frame):
        """
        Process mask data for visualization.
        
        Args:
            mask: Mask data from neural network (can be None, list, or tensor)
            frame: Current frame number for debug logging
            in_width: Width of input frame
            in_height: Height of input frame
            display_frame: Frame to overlay heatmap onto
            
        Returns:
            tuple: (processed_image, mask_total) - Image with heatmap and the processed mask
        """
        print(f"[DEBUG] Frame {frame}: Processing mask (type: {type(mask)})")
        try:
            mask_captures = None
            mask_total = None
            
            # Handle different mask types
            if mask is None:
                print(f"[DEBUG] Frame {frame}: No mask available")
                return None, None
                
            elif isinstance(mask, list) and len(mask) > 0:
                # Process list of masks
                print(f"[DEBUG] Frame {frame}: Mask list length: {len(mask)}")
                mask_captures = []
                for i, m in enumerate(mask):
                    print(f"[DEBUG] Frame {frame}: Processing mask item {i}, shape: {m.shape}")
                    mask_captures.append(m[:, :1].clone())
                mask_total = sum(mask_captures)
                print(f"[DEBUG] Frame {frame}: Mask processing complete")
                
            elif isinstance(mask, torch.Tensor):
                # Handle tensor mask
                print(f"[DEBUG] Frame {frame}: Processing tensor mask, shape: {mask.shape}")
                # Take just the first sample if batch size > 1
                if len(mask.shape) > 3:
                    mask = mask[0]
                # If multi-channel, use the mean across channels
                if len(mask.shape) > 2 and mask.shape[0] > 1:
                    mask_total = torch.mean(mask, dim=0, keepdim=True)
                else:
                    mask_total = mask
                print(f"[DEBUG] Frame {frame}: Tensor mask processing complete")
                
            else:
                print(f"[DEBUG] Frame {frame}: Unsupported mask type")
                return None, None
            
            # Create heatmap visualization if mask exists
            if mask_total is not None:
                try:
                    print(f"[DEBUG] Frame {frame}: Creating heatmap visualization")
                    # Convert mask to numpy and rescale to 0-255
                    with torch.no_grad():
                        mask_np = mask_total.cpu().numpy()[0]  # Remove channel dimension
                        
                        # Normalize the mask to 0-1 range
                        mask_min = mask_np.min()
                        mask_max = mask_np.max()
                        if mask_max > mask_min:
                            mask_np = (mask_np - mask_min) / (mask_max - mask_min)
                        
                        # Resize mask to match image dimensions
                        mask_np = cv2.resize(mask_np, (in_width, in_height))
                        
                        # Apply colormap (jet is common for heatmaps)
                        heatmap = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        
                        # Convert BGR to RGB (OpenCV uses BGR)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        
                        # Convert display frame to numpy array for overlay
                        display_np = display_frame.cpu().numpy()[0]  # Get first channel
                        display_np = cv2.resize(display_np, (in_width, in_height))
                        
                        # Create a 3-channel image from grayscale
                        display_rgb = np.stack([display_np, display_np, display_np], axis=2)
                        
                        # Overlay the heatmap with alpha blending
                        alpha = 0.5  # Transparency level
                        overlay = cv2.addWeighted(
                            (display_rgb * 255).astype(np.uint8), 1-alpha,
                            heatmap, alpha, 0
                        )
                        
                        # Return the overlaid image
                        return Image.fromarray(overlay), mask_total
                        
                except Exception as e:
                    print(f"[ERROR] Frame {frame}: Failed to create heatmap: {e}")
            
            # Return original display frame if no visualization was created
            cur_image = transforms.ToPILImage()(display_frame.cpu())
            return cur_image, mask_total
            
        except Exception as e:
            print(f"[ERROR] Frame {frame}: Error processing mask: {e}")
            # Continue with empty masks
            cur_image = transforms.ToPILImage()(display_frame.cpu())
            return cur_image, None


    def create_info_panel(self, frame, in_width, in_height, network_output, true_label):
        """Create the information panel with predictions and labels."""
        with torch.no_grad():
            # Apply post-processing based on loss function
            prediction = nn_postprocess(network_output)
            pred_class = torch.argmax(prediction, dim=1).item()
            confidence = prediction[0, pred_class].item()
        
        # Add safe class name lookup
        class_name = args.class_names[pred_class] if pred_class < len(args.class_names) else f"Unknown ({pred_class})"
        
        # Create a new blank image for the info panel
        info_width = in_width // 3
        info_panel = Image.new('L', (info_width, in_height), 0)
        draw = ImageDraw.Draw(info_panel)
        
        # Add title and frame number
        draw.text((10, 10), f"Frame: {frame}", fill=255)
        draw.text((10, 40), f"Prediction: {class_name}", fill=255)
        draw.text((10, 70), f"Confidence: {confidence:.2f}", fill=255)
        draw.text((10, 100), f"True label: {true_label}", fill=255)
        
        return info_panel, pred_class, confidence

    def cleanup_resources(self, tensors_to_delete, sample_frames):
        """Clean up memory resources to prevent memory leaks."""
        # Delete specific tensors
        for tensor in tensors_to_delete:
            del tensor
        
        # Remove all frames and clear the list
        for tensor in sample_frames:
            del tensor
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return []  # Return empty list for sample_frames

    def create_output_video(self, temp_dir, frame_count):
        """Create the output video from saved frames."""
        if frame_count > 0:
            print(f"[DEBUG] Creating video from {frame_count} saved frames")
            frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            # Sanitize output filename and enclose in quotes
            safe_output = f'"{self.output_name.replace(" ", "_")}"'
            
            ffmpeg_cmd = f"ffmpeg -y -framerate 30 -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf 23 {safe_output}"
            print(f"[DEBUG] Running command: {ffmpeg_cmd}")
            
            # Run command and check for errors
            exit_code = os.system(ffmpeg_cmd)
            if exit_code != 0:
                print(f"[ERROR] FFmpeg failed with exit code {exit_code}")
                return False
                
            print(f"[DEBUG] Video saved to {self.output_name}")
            return True
        else:
            print("[ERROR] No frames were processed!")
            return False
    
    def run_neural_network(self, sample_frames, frame):
        """Process frames through the neural network to get predictions and masks."""
        print(f"[DEBUG] Frame {frame}: Processing batch of {len(sample_frames)} frames")
        
        with torch.no_grad():
            # Concatenate frames into input tensor
            image_input = torch.cat(sample_frames, 1)
            
            # Normalize the input
            m = torch.mean(image_input)
            v = torch.std(image_input)
            net_input = (image_input - m) / v
            
            # Forward pass through the network
            print(f"[DEBUG] Frame {frame}: Running forward pass")
            out, mask = self.net.vis_forward(net_input)
            print(f"[DEBUG] Frame {frame}: Network forward pass complete")
    
        return out, mask, net_input

    def process_video(self):
        print(f"[DEBUG] Starting video processing for {self.path}")
        print(f"[DEBUG] Target frames: {self.begin_frame} to {self.end_frame}")

        # Create a directory for temporary frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        CHUNK_SIZE = 500
        
        # Create the video decoder and initialize variables
        decoder = EfficientVideoDecoder(
            video_path=self.path,
            width=self.width,
            height=self.height,
            scale=self.scale,
            crop_x=self.crop_x,
            crop_y=self.crop_y,
            begin_frame=self.begin_frame,
            end_frame=self.end_frame,
            frames_per_sample=self.frames_per_sample,
            planes = 1,
            src = 'RGB'
        )
        
        in_width, in_height = decoder.in_width, decoder.in_height
        frame = self.begin_frame
        sample_frames = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        frame_count = 0
        
        try:
            # Process frames in chunks
            for chunk_start in range(self.begin_frame, self.end_frame, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, self.end_frame)
                print(f"Processing chunk {chunk_start}-{chunk_end} ({100.0*chunk_start/self.end_frame:.1f}%)")
                self.cleanup_resources([], [])  # Just run GC
                
                while frame < chunk_end:
                    # Progress reporting
                    if frame % 100 == 0:
                        print(f"Processing frame {frame}/{self.end_frame} ({frame*100.0/self.end_frame:.1f}%)")
                    
                    # Read and process frame
                    in_frame = decoder.read_frame()
                    if in_frame is None:
                        print("Failed to read - End of video reached")
                        break
                        
                    # Add frame to sample
                    sample_frames.append(in_frame.to(device=device, dtype=torch.float))
                    
                    # Process when we have enough frames
                    if len(sample_frames) == self.frames_per_sample:
                        # Run neural network
                        out, mask, net_input = self.run_neural_network(sample_frames, frame)
                        
                        # Process mask and create heatmap
                        display_frame = sample_frames[-1][0]
                        cur_image, mask_total = self.process_mask(mask, frame, in_width, in_height, display_frame)
                        
                        # Create info panel
                        true_label = self.video_labels.getLabel(frame)
                        info_panel, pred_class, confidence = self.create_info_panel(frame, in_width, in_height, out, true_label)
                        
                        # Combine image with info panel
                        if cur_image.mode != 'RGB':
                            cur_image = cur_image.convert('RGB')
                        
                        # Create padded image with info panel
                        info_width = in_width // 3
                        padded_image = ImageOps.pad(cur_image, (in_width + info_width, in_height), centering=(0,0))
                        
                        # Add info to padded image
                        draw = ImageDraw.Draw(padded_image)
                        # Save frame
                        frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                        padded_image.save(frame_filename)
                        frame_count += 1
                        
                        # Debug first frame
                        if frame == self.begin_frame:
                            debug_filename = os.path.join(temp_dir, "debug_first_frame.png")
                            padded_image.save(debug_filename)
                            print(f"Saved debug frame to {debug_filename}")
                        
                        # Cleanup
                        tensors_to_delete = [net_input, out, mask, mask_total, display_frame]
                        sample_frames = self.cleanup_resources(tensors_to_delete, sample_frames)
                    
                    # Update frame counter
                    frame += 1
                
                # Force garbage collection after each chunk
                self.cleanup_resources([], [])
                    
            # Create output video
            self.create_output_video(temp_dir, frame_count)
                    
        finally:
            # Clean up
            decoder.release()
            print(f"[DEBUG] Video processing complete")


image_size = (args.dnn_channels * args.frames_per_sample, args.height, args.width)

# Use device rather than cuda(device) because of the use case of running on cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

# Model setup stuff
if 'alexnet' == args.modeltype:
    net = AlexLikeNet(**checkpoint["metadata"]["model_args"],)
elif 'resnet18' == args.modeltype:
    net = ResNet18(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True)
elif 'resnet34' == args.modeltype:
    net = ResNet34(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True)
elif 'bennet' == args.modeltype:
    net = BenNet(in_dimensions=image_size, out_classes=args.label_classes)
elif 'resnext50' == args.modeltype:
    net = ResNext50(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True)
elif 'resnext34' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext34(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=False,
            use_dropout=False)
elif 'resnext18' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext18(in_dimensions=image_size, out_classes=args.label_classes, expanded_linear=True,
            use_dropout=False)
elif 'convnextxt' == args.modeltype:
    net = ConvNextExtraTiny(in_dimensions=image_size, out_classes=args.label_classes)
elif 'convnextt' == args.modeltype:
    net = ConvNextTiny(in_dimensions=image_size, out_classes=args.label_classes)
elif 'convnexts' == args.modeltype:
    net = ConvNextSmall(in_dimensions=image_size, out_classes=args.label_classes)
elif 'convnextb' == args.modeltype:
    net = ConvNextBase(in_dimensions=image_size, out_classes=args.label_classes)

net = net.to(device)
print(f"Model is {net}")

# See if the model weights can be restored.
if args.resume_from is not None:
    restoreModel(args.resume_from, net, device)

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
    file_col = header.index('filename')
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
