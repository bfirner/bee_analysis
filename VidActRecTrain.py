#! /usr/bin/python3

"""
This will train a model using a webdataset tar archive for data input.
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
import torch
import webdataset as wds
from collections import namedtuple
# Helper function to convert to images
from torchvision import transforms

from models.alexnet import AlexLikeNet
from models.resnet import (ResNet18, ResNet34)


parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set.")
parser.add_argument(
    'dataset',
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
    default=50,
    help='Total epochs to train.')
parser.add_argument(
    '--seed',
    type=int,
    required=False,
    default='0',
    help="Seed to use for RNG initialization.")
parser.add_argument(
    '--modeltype',
    type=str,
    required=False,
    default="alexnet",
    choices=["alexnet", "resnet18", "resnet34"],
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

args = parser.parse_args()

in_frames = args.sample_frames
decode_strs = []
# The image for a particular frame
for i in range(in_frames):
    decode_strs.append(f"{i}.png")
# The class label
decode_strs.append("cls")
# Metadata for this sample. A string of format: f"{video_path},{frame},{time}"
decode_strs.append("metadata.txt")

# Decode the proper number of items for each sample from the dataloader
# The field names are just being taken from the decode strings, but they cannot begin with a digit
# or contain the '.' character, so the character "f" is prepended to each string and the '.' is
# replaced with a '_'. The is a bit egregious, but it does guarantee that the tuple being used to
# accept the output of the dataloader matches what is being used in webdataset decoding.
LoopTuple = namedtuple('LoopTuple', ' '.join(["f" + s for s in decode_strs]).replace('.', '_'))
dl_tuple = LoopTuple(*([None] * len(decode_strs)))

# TODO FIXME Deterministic shuffle, only shuffles within a range. Should manipulate what is in the
# tar file, shuffling after the dataset is created.
dataset = (
    wds.WebDataset(args.dataset)
    .shuffle(20000, initial=20000)
    .decode("l")
    .to_tuple(*decode_strs)
)

dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=32)

if args.evaluate:
    eval_dataset = (
        wds.WebDataset(args.evaluate)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=0, batch_size=32)


# Hard coding the Alexnet like network for now.
# TODO Also hard coding the input and output sizes
if 'alexnet' == args.modeltype:
    net = AlexLikeNet(in_dimensions=(in_frames, 400, 400), out_classes=3, linear_size=512).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4)
elif 'resnet18' == args.modeltype:
    net = ResNet18(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=True).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-5)
elif 'resnet34' == args.modeltype:
    net = ResNet34(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=True).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
print(f"Model is {net}")

# See if the model weights and optimizer state should be restored.
if args.resume_from is not None:
    checkpoint = torch.load(args.resume_from)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])

train_as_classifier = True
if train_as_classifier:
    #total = 11. + 16. + 12.
    #loss_fn = torch.nn.NLLLoss(weight=torch.tensor([total/11., total/16., total/12.]).cuda())
    loss_fn = torch.nn.NLLLoss()
else:
    #loss_fn = torch.nn.L1Loss()
    loss_fn = torch.nn.MSELoss()

if not args.no_train:
    for epoch in range(args.epochs):
        # Make a confusion matrix, x is item, y is classification
        totals=[[0,0,0],
                [0,0,0],
                [0,0,0]]
        print(f"Starting epoch {epoch}")
        for batch_num, dl_tuple in enumerate(dataloader):
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
                net_input = torch.cat(*(raw_input), dim=1)

            out = net.forward(net_input)

            # Convert the labels to a one hot encoding to serve at the DNN target.
            # The label class is 1 based, but need to be 0-based for the one_hot function.
            labels = dl_tuple[-2]
            if train_as_classifier:
                loss = loss_fn(out, labels.cuda() - 1)
            else:
                labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                loss = loss_fn(out, labels.cuda().float() - 1)
            loss.backward()
            optimizer.step()
            # Fill in the confusion matrix
            with torch.no_grad():
                classes = torch.argmax(out, dim=1)
                if train_as_classifier:
                    labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                for i in range(labels.size(0)):
                    for j in range(3):
                        # If this is the j'th class
                            if 1 == labels[i][j]:
                                totals[j][classes[i]] += 1
        print(f"Finished epoch {epoch}, last loss was {loss}")
        print(f"Confusion matrix:")
        print(totals)
        accuracy = (totals[0][0] + totals[1][1] + totals[2][2])/(sum(totals[0]) + sum(totals[1]) + sum(totals[2]))
        print(f"Accuracy: {accuracy}")
        # Validation set
        if args.evaluate is not None:
            net.eval()
            with torch.no_grad():
                # Make a confusion matrix, x is item, y is classification
                totals=[[0,0,0],
                        [0,0,0],
                        [0,0,0]]
                for batch_num, dl_tuple in enumerate(eval_dataloader):
                    if 1 == in_frames:
                        net_input = dl_tuple[0].unsqueeze(1).cuda()
                    else:
                        raw_input = []
                        for i in range(in_frames):
                            raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                        net_input = torch.cat(*(raw_input), dim=1)
                    out = net.forward(net_input)
                    labels = dl_tuple[-2]
                    # Convert the labels to a one hot encoding to serve at the DNN target.
                    # The label class is 1 based, but need to be 0-based for the one_hot function.
                    if train_as_classifier:
                        loss = loss_fn(out, labels.cuda() - 1)
                    else:
                        labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                        loss = loss_fn(out, labels.cuda().float() - 1)
                    with torch.no_grad():
                        classes = torch.argmax(out, dim=1)
                        if train_as_classifier:
                            labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                        for i in range(labels.size(0)):
                            for j in range(3):
                                # If this is the j'th class
                                    if 1 == labels[i][j]:
                                        totals[j][classes[i]] += 1
                # Print evaluation information
                print(f"Evaluation confusion matrix:")
                print(totals)
                accuracy = (totals[0][0] + totals[1][1] + totals[2][2])/(sum(totals[0]) + sum(totals[1]) + sum(totals[2]))
                print(f"Accuracy: {accuracy}")
            net.train()

    torch.save({
        "model_dict": net.state_dict(),
        "optim_dict": optimizer.state_dict()}, args.outname)

# Post-training evaluation
if args.evaluate is not None:
    print("Evaluating model.")
    net.eval()
    with torch.no_grad():
        # Make a confusion matrix, x is item, y is classification
        totals=[[0,0,0],
                [0,0,0],
                [0,0,0]]
        for batch_num, dl_tuple in enumerate(eval_dataloader):
            # Decoding only the luminance channel means that the channel dimension has gone away here.
            if 1 == in_frames:
                net_input = dl_tuple[0].unsqueeze(1).cuda()
            else:
                raw_input = []
                for i in range(in_frames):
                    raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                net_input = torch.cat(*(raw_input), dim=1)
            out = net.forward(net_input)
            # Convert the labels to a one hot encoding to serve at the DNN target.
            # The label class is 1 based, but need to be 0-based for the one_hot function.
            labels = dl_tuple[-2]
            if train_as_classifier:
                loss = loss_fn(out, labels.cuda() - 1)
            else:
                labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                loss = loss_fn(out, labels.cuda().float() - 1)
            with torch.no_grad():
                classes = torch.argmax(out, dim=1)
                if train_as_classifier:
                    labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                for i in range(labels.size(0)):
                    for j in range(3):
                        # If this is the j'th class
                            if 1 == labels[i][j]:
                                totals[j][classes[i]] += 1
        # Print evaluation information
        print(f"Evaluation confusion matrix:")
        print(totals)
        accuracy = (totals[0][0] + totals[1][1] + totals[2][2])/(sum(totals[0]) + sum(totals[1]) + sum(totals[2]))
        print(f"Accuracy: {accuracy}")
