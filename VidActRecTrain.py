#! /usr/bin/python3

"""
This will train a model using a webdataset tar archive for data input.
If you use Cuda >= 10.2 then running in deterministic mode requires this environment variable to be
set: CUBLAS_WORKSPACE_CONFIG=:4096:8
To disable deterministic mode use the option --not_deterministic
There is another problem is AveragePooling that may also make it impossible to use deterministic
mode. If you encounter it, the solution is to just disable determinism with --not_deterministic
"""

# Not using video reading library from torchvision.
# It only works with old versions of ffmpeg.
import argparse
import csv
import datetime
import functools
import heapq
import io
import math
import numpy
import os
import random
import sys
import torch
import torch.cuda.amp
import webdataset as wds
from collections import namedtuple
# Helper function to convert to images
from torchvision import transforms

from utility.dataset_utility import (extractLabels, getImageSize, getLabelSize)
from utility.eval_utility import (ConfusionMatrix, RegressionResults, WorstExamples)
from utility.model_utility import (restoreModelAndState)
from utility.train_utility import (updateWithScaler, updateWithoutScaler)

from models.alexnet import AlexLikeNet
from models.bennet import BenNet
from models.resnet import (ResNet18, ResNet34)
from models.resnext import (ResNext18, ResNext34, ResNext50)
from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)



# Argument parser setup for the program.
parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set.")
parser.add_argument(
    '--template',
    required=False,
    default=None,
    choices=['bees', 'multilabel_detection'],
    type=str,
    help=('Set other options automatically based upon a typical training template.'
        'Template settings are overriden by other selected options.'
        'bees: Alexnet model with index labels are converted to one hot labels.'
        'multilabel: Multilabels are loaded from "detection.pth", binary cross entropy loss is used.'))
parser.add_argument(
    'dataset',
    nargs='+',
    type=str,
    help='Dataset for training.')
parser.add_argument(
    '--sample_frames',
    type=int,
    required=False,
    default=1,
    help='Number of frames in each sample.')
parser.add_argument(
    '--outname',
    type=str,
    required=False,
    default="model.checkpoint",
    help='Base name for model and checkpoint saving.')
parser.add_argument(
    '--resume_from',
    type=str,
    required=False,
    help='Model weights to restore.')
parser.add_argument(
    '--epochs',
    type=int,
    required=False,
    default=15,
    help='Total epochs to train.')
parser.add_argument(
    '--seed',
    type=int,
    required=False,
    default='0',
    help="Seed to use for RNG initialization.")
parser.add_argument(
    '--normalize',
    required=False,
    default=False,
    action="store_true",
    help=("Normalize inputs: input = (input - mean) / stddev. "
        "Note that VidActRecDataprep is already normalizing so this may not be required."))
parser.add_argument(
    '--modeltype',
    type=str,
    required=False,
    default="resnext18",
    choices=["alexnet", "resnet18", "resnet34", "bennet", "resnext50", "resnext34", "resnext18",
    "convnextxt", "convnextt", "convnexts", "convnextb"],
    help="Model to use for training.")
parser.add_argument(
    '--no_train',
    required=False,
    default=False,
    action='store_true',
    help='Set this flag to skip training. Useful to load an already trained model for evaluation.')
parser.add_argument(
    '--evaluate',
    type=str,
    required=False,
    default=None,
    help='Evaluate with the given dataset.')
parser.add_argument(
    '--save_top_n',
    type=int,
    required=False,
    default=None,
    help='Save N images for each class with the highest classification score. Works with --evaluate')
parser.add_argument(
    '--save_worst_n',
    type=int,
    required=False,
    default=None,
    help='Save N images for each class with the lowest classification score. Works with --evaluate')
parser.add_argument(
    '--not_deterministic',
    required=False,
    default=False,
    action='store_true',
    help='Set this to disable deterministic training.')
parser.add_argument(
    '--labels',
    type=str,
    # Support an array of strings to have multiple different label targets.
    nargs='+',
    required=False,
    default=["cls"],
    help='File to decode from webdataset as the DNN output target labels.')
parser.add_argument(
    '--skip_metadata',
    required=False,
    default=False,
    action='store_true',
    help='Set to skip loading metadata.txt from the webdataset.')
parser.add_argument(
    '--convert_idx_to_classes',
    required=False,
    default=1,
    choices=[0, 1],
    type=int,
    help='Train output as a classifier by converting all labels to one-hot vectors.')
parser.add_argument(
    '--num_outputs',
    required=False,
    default=3,
    type=int,
    help='The elements of the one-hot encoding of the label, if convert_idx_to_classes is set.')
parser.add_argument(
    '--label_offset',
    required=False,
    default=1,
    type=int,
    help='The starting value of classes when training with cls labels (the labels value is "cls").')
parser.add_argument(
    '--loss_fun',
    required=False,
    default='CrossEntropyLoss',
    choices=['NLLLoss', 'BCEWithLogitsLoss', 'CrossEntropyLoss', 'L1Loss', 'MSELoss', 'BCELoss'],
    type=str,
    help="Loss function to use during training.")

args = parser.parse_args()

# these next clauses are for scripts so that we have a log of what was called on what machine and when.
python_log =  os.system("which python3")
machine_log = os.system("uname -a")
date_log = os.system("date")

print("Log: Program_args: ",end='')
for theArg in sys.argv :
    print(theArg + " ",end='')
print(" ")

print("Log: Started: ",date_log)
print("Log: Machine: ",machine_log)
print("Log: Python_version: ",python_log)

torch.use_deterministic_algorithms(not args.not_deterministic)
torch.manual_seed(args.seed)
random.seed(0)
numpy.random.seed(0)

if args.template is not None:
    if 'bees' == args.template:
        if '--modeltype' not in sys.argv:
            args.modeltype = 'alexnet'
    elif 'multilabel_detection' == args.template:
        if '--modeltype' not in sys.argv:
            args.modeltype = 'bennet'
        if '--skip_metadata' not in sys.argv:
            args.skip_metadata = True
        if '--convert_idx_to_classes' not in sys.argv:
            args.convert_idx_to_classes = 0
        if '--labels' not in sys.argv:
            args.labels = ['detection.pth']
        if '--loss_fun' not in sys.argv:
            args.loss_fun = 'BCEWithLogitsLoss'

# Convert the numeric input to a bool
convert_idx_to_classes = args.convert_idx_to_classes == 1

# Network outputs may need to be postprocessed for evaluation if some postprocessing is being done
# automatically by the loss function.
if 'CrossEntropyLoss' == args.loss_fun:
    nn_postprocess = torch.nn.Softmax(dim=1)
elif 'BCEWithLogitsLoss' == args.loss_fun:
    nn_postprocess = torch.nn.Sigmoid()
else:
    # Otherwise just use an identify function.
    nn_postprocess = lambda x: x


loss_fn = getattr(torch.nn, args.loss_fun)()

# Later on we will need to change behavior if the loss function is regression rather than
# classification
regression_loss = ['L1Loss', 'MSELoss']

in_frames = args.sample_frames
decode_strs = []
# The image for a particular frame
for i in range(in_frames):
    decode_strs.append(f"{i}.png")
# The class label(s)
label_range = slice(len(decode_strs), len(decode_strs) + len(args.labels))
for label_str in args.labels:
    decode_strs.append(label_str)
# Metadata for this sample. A string of format: f"{video_path},{frame},{time}"
if not args.skip_metadata:
    metadata_index = len(decode_strs)
    decode_strs.append("metadata.txt")

# The default labels for the bee videos are "1, 2, 3" instead of "0, 1, 2"
if "cls" != args.labels[0]:
    label_offset = 0
else:
    label_offset = args.label_offset


print(f"Training with dataset {args.dataset}")
# If we are converting to a one-hot encoding output then we need to check the argument that
# specifies the number of output elements. Otherwise we can check the number of elements in the
# webdataset.
if convert_idx_to_classes:
    label_size = args.num_outputs * getLabelSize(args.dataset, decode_strs, label_range)
else:
    label_size = getLabelSize(args.dataset, decode_strs, label_range)

# Decode the proper number of items for each sample from the dataloader
# The field names are just being taken from the decode strings, but they cannot begin with a digit
# or contain the '.' character, so the character "f" is prepended to each string and the '.' is
# replaced with a '_'. The is a bit egregious, but it does guarantee that the tuple being used to
# accept the output of the dataloader matches what is being used in webdataset decoding.
LoopTuple = namedtuple('LoopTuple', ' '.join(["f" + s for s in decode_strs]).replace('.', '_'))
dl_tuple = LoopTuple(*([None] * len(decode_strs)))

# TODO FIXME Deterministic shuffle only shuffles within a range. Should perhaps manipulate what is
# in the tar file by shuffling filenames after the dataset is created.
dataset = (
    wds.WebDataset(args.dataset, shardshuffle=True)
    .shuffle(20000//in_frames, initial=20000//in_frames)
    .decode("l")
    .to_tuple(*decode_strs)
)

image_size = getImageSize(args.dataset, decode_strs)
print(f"Decoding images of size {image_size}")

batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size)

if args.evaluate:
    eval_dataset = (
        wds.WebDataset(args.evaluate)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=0, batch_size=batch_size)


lr_scheduler = None
# AMP doesn't seem to like all of the different model types, so disable it unless it has been
# verified.
use_amp = False
if 'alexnet' == args.modeltype:
    net = AlexLikeNet(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size, linear_size=512).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,7], gamma=0.2)
    use_amp = True
elif 'resnet18' == args.modeltype:
    net = ResNet18(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size, expanded_linear=True).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-5)
elif 'resnet34' == args.modeltype:
    net = ResNet34(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size, expanded_linear=True).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
elif 'bennet' == args.modeltype:
    net = BenNet(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
elif 'resnext50' == args.modeltype:
    net = ResNext50(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size, expanded_linear=True).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3], gamma=0.1)
    batch_size = 64
elif 'resnext34' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext34(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size, expanded_linear=False,
            use_dropout=False).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,9], gamma=0.2)
elif 'resnext18' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext18(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size, expanded_linear=True,
            use_dropout=False).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextxt' == args.modeltype:
    net = ConvNextExtraTiny(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4, weight_decay=10e-4, momentum=0.9,
            nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5,12], gamma=0.2)
    use_amp = True
elif 'convnextt' == args.modeltype:
    net = ConvNextTiny(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnexts' == args.modeltype:
    net = ConvNextSmall(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextb' == args.modeltype:
    net = ConvNextBase(in_dimensions=(in_frames, image_size[1], image_size[2]), out_classes=label_size).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
print(f"Model is {net}")

# See if the model weights and optimizer state should be restored.
if args.resume_from is not None:
    restoreModelAndState(args.resume_from, net, optimizer)

# Gradient scaler for mixed precision training
if use_amp:
    scaler = torch.cuda.amp.GradScaler()

# TODO(bfirner) Read class names from something
class_names = []
for i in range(label_size):
    class_names.append(f"{i}")

if not args.no_train:
    worst_training = None
    if args.save_worst_n is not None:
        worst_training = WorstExamples(
            args.outname.split('.')[0] + "-worstN-train", class_names, args.save_worst_n)
        print(f"Saving {args.save_worst_n} highest error training images to {worst_training.worstn_path}.")
    for epoch in range(args.epochs):
        # Make a confusion matrix or loss statistics
        if args.loss_fun in regression_loss:
            totals = RegressionResults(size=label_size)
        else:
            totals = ConfusionMatrix(size=label_size)
        print(f"Starting epoch {epoch}")
        for batch_num, dl_tuple in enumerate(dataloader):
            dateNow = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            if ( (batch_num % 1000) == 1):
                print ("Log: at tuple %d at %s" % (batch_num,dateNow))

            optimizer.zero_grad()
            # For debugging purposes
            # img = transforms.ToPILImage()(dl_tuple[0][0]).convert('RGB')
            # img.save(f"batch_{batch_num}.png")
            # Decoding only the luminance channel means that the channel dimension has gone away here.
            if 1 == in_frames:
                net_input = dl_tuple[0].unsqueeze(1).cuda()
            else:
                raw_input = []
                for i in range(in_frames):
                    raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                net_input = torch.cat(raw_input, dim=1)
            # Normalize inputs: input = (input - mean)/stddev
            if args.normalize:
                v, m = torch.var_mean(net_input)
                net_input = (net_input - m) / v

            labels = extractLabels(dl_tuple,label_range).cuda()

            # The label value may need to be adjusted, for example if the label class is 1 based, but
            # should be 0-based for the one_hot function.
            labels = labels - label_offset

            if use_amp:
                out, loss = updateWithScaler(loss_fn, net, net_input, labels, scaler, optimizer)
            else:
                out, loss = updateWithoutScaler(loss_fn, net, net_input, labels, optimizer)

            # Fill in the confusion matrix and worst examples.
            with torch.no_grad():
                if args.loss_fun in regression_loss:
                    post_out = nn_postprocess(out)
                    totals.update(predictions=post_out, labels=labels)
                    # TODO FIXME Worst and best eval for regression outputs
                else:
                    # The postprocessesing should include Softmax or similar if that is required for
                    # the network. Outputs of most classification networks are considered
                    # probabilities (but only take that in a very loose sense of the word) so
                    # rounding could be appropriate.
                    classes = nn_postprocess(out)
                    # Convert index labels to a one hot encoding for standard processing.
                    if convert_idx_to_classes:
                        labels = torch.nn.functional.one_hot(labels, num_classes=label_size)
                    totals.update(predictions=classes, labels=labels)
                    if worst_training is not None:
                        if args.skip_metadata:
                            metadata = [""] * labels.size(0)
                        else:
                            metadata = dl_tuple[metadata_index]
                        # For each item in the batch see if it requires an update to the worst examples
                        # If the DNN should have predicted this image was a member of the labelled class
                        # then see if this image should be inserted into the worst_n queue for the
                        # labelled class based upon the DNN output for this class.
                        input_images = dl_tuple[0]
                        for i in range(labels.size(0)):
                            label = torch.argwhere(labels[i])[0].item()
                            worst_training.test(label, out[i][label].item(), input_images[i], metadata[i])
        print(f"Finished epoch {epoch}, last loss was {loss}")
        print(f"Training results:")
        print(totals.makeResults())
        if worst_training is not None:
            worst_training.save(epoch)
        # Validation set
        if args.evaluate is not None:
            net.eval()
            with torch.no_grad():
                # Make a confusion matrix or loss statistics
                if args.loss_fun in regression_loss:
                    totals = RegressionResults(size=label_size)
                else:
                    totals = ConfusionMatrix(size=label_size)
                for batch_num, dl_tuple in enumerate(eval_dataloader):
                    if 1 == in_frames:
                        net_input = dl_tuple[0].unsqueeze(1).cuda()
                    else:
                        raw_input = []
                        for i in range(in_frames):
                            raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                        net_input = torch.cat(raw_input, dim=1)
                    # Normalize inputs: input = (input - mean)/stddev
                    if args.normalize:
                        v, m = torch.var_mean(net_input)
                        net_input = (net_input - m) / v

                    with torch.cuda.amp.autocast():
                        out = net.forward(net_input)
                        labels = extractLabels(dl_tuple,label_range).cuda()

                        # The label value may need to be adjusted, for example if the label class is
                        # 1-based, but should be 0-based for the one_hot function.
                        labels = labels-label_offset

                        loss = loss_fn(out, labels.cuda())
                    with torch.no_grad():
                        if args.loss_fun in regression_loss:
                            post_out = nn_postprocess(out)
                            totals.update(predictions=post_out, labels=labels)
                        else:
                            # The postprocessesing should include Softmax or similar if that is required for
                            # the network. Outputs of most classification networks are considered
                            # probabilities (but only take that in a very loose sense of the word) so
                            # rounding is appropriate.
                            # classes = torch.round(nn_postprocess(out)).clamp(0, 1)
                            classes = nn_postprocess(out)
                            # Convert index labels to a one hot encoding for standard processing.
                            if convert_idx_to_classes:
                                labels = torch.nn.functional.one_hot(labels, num_classes=label_size)
                            totals.update(predictions=classes, labels=labels)
                # Print evaluation information
                print(f"Evaluation results:")
                print(totals.makeResults())
            net.train()
        # Adjust learning rate according to the learning rate schedule
        if lr_scheduler is not None:
            lr_scheduler.step()

    torch.save({
        "model_dict": net.state_dict(),
        "optim_dict": optimizer.state_dict(),
        "py_random_state": random.getstate(),
        "np_random_state": numpy.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        }, args.outname)

# Post-training evaluation
if args.evaluate is not None:
    top_eval = None
    worst_eval = None
    print("Evaluating model.")
    if args.save_top_n is not None:
        top_eval = WorstExamples(
            args.outname.split('.')[0] + "-topN-eval", class_names, args.save_top_n,
            worst_mode=False)
        print(f"Saving {args.save_top_n} highest error evaluation images to {top_eval.worstn_path}.")
    if args.save_worst_n is not None:
        worst_eval = WorstExamples(
            args.outname.split('.')[0] + "-worstN-eval", class_names, args.save_worst_n)
        print(f"Saving {args.save_worst_n} highest error evaluation images to {worst_eval.worstn_path}.")

    net.eval()
    with torch.no_grad():
        # Make a confusion matrix or loss statistics
        if args.loss_fun in regression_loss:
            totals = RegressionResults(size=label_size)
        else:
            totals = ConfusionMatrix(size=label_size)
        with open(args.outname.split('.')[0] + ".log", 'w') as logfile:
            logfile.write('video_path,frame,time,label,prediction\n')
            for batch_num, dl_tuple in enumerate(eval_dataloader):
                # Decoding only the luminance channel means that the channel dimension has gone away here.
                if 1 == in_frames:
                    net_input = dl_tuple[0].unsqueeze(1).cuda()
                else:
                    raw_input = []
                    for i in range(in_frames):
                        raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                    net_input = torch.cat(raw_input, dim=1)
                # Normalize inputs: input = (input - mean)/stddev
                if args.normalize:
                    v, m = torch.var_mean(net_input)
                    net_input = (net_input - m) / v

                # Visualization masks are not supported with all model types yet.
                if args.modeltype in ['alexnet', 'bennet', 'resnet18', 'resnet34']:
                    out, mask = net.vis_forward(net_input)
                else:
                    out = net.forward(net_input)
                    mask = [None] * batch_size
                # Convert the labels to a one hot encoding to serve at the DNN target.
                # The label class is 1 based, but need to be 0-based for the one_hot function.
                labels = extractLabels(dl_tuple,label_range).cuda()

                # The label value may need to be adjusted, for example if the label class is 1 based, but
                # should be 0-based for the one_hot function.
                labels = labels-label_offset

                if args.skip_metadata:
                    metadata = [""] * labels.size(0)
                else:
                    metadata = dl_tuple[metadata_index]

                loss = loss_fn(out, labels.cuda())

                with torch.no_grad():
                    if args.loss_fun in regression_loss:
                        post_out = nn_postprocess(out)
                        totals.update(predictions=post_out, labels=labels)
                        # TODO FIXME Worst and best eval for regression outputs
                    else:
                        # The postprocessesing should include Softmax or similar if that is required for
                        # the network. Outputs of most classification networks are considered
                        # probabilities (but only take that in a very loose sense of the word) so
                        # rounding is appropriate.
                        # classes = torch.round(nn_postprocess(out)).clamp(0, 1)
                        classes = nn_postprocess(out)
                        # Convert index labels to a one hot encoding for standard processing.
                        if convert_idx_to_classes:
                            labels = torch.nn.functional.one_hot(labels, num_classes=label_size)
                        totals.update(predictions=classes, labels=labels)
                        classes = torch.round(classes).clamp(0, 1)
                        # Log the predictions
                        for i in range(labels.size(0)):
                            logfile.write(','.join((metadata[i], str(out[i]), str(labels[i]))))
                            logfile.write('\n')
                        if worst_eval is not None or top_eval is not None:
                            # For each item in the batch see if it requires an update to the worst examples
                            # If the DNN should have predicted this image was a member of the labelled class
                            # then see if this image should be inserted into the worst_n queue for the
                            # labelled class based upon the DNN output for this class.
                            input_images = dl_tuple[0]
                            for i in range(labels.size(0)):
                                label = torch.argwhere(labels[i])[0].item()
                                if worst_eval is not None:
                                    worst_eval.test(label, out[i][label].item(), input_images[i], metadata[i])
                                if top_eval is not None:
                                    top_eval.test(label, out[i][label].item(), input_images[i], metadata[i])

        # Save the worst and best examples
        if worst_eval is not None:
            worst_eval.save("evaluation")
        if top_eval is not None:
            top_eval.save("evaluation")

        # Print evaluation information
        print(f"Evaluation results:")
        print(totals.makeResults())
