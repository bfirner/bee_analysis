#! /usr/bin/python3

"""
This will train a model using a webdataset tar archive for data input.
If you use Cuda >= 10.2 then running in deterministic mode requires this environment variable to be
set: CUBLAS_WORKSPACE_CONFIG=:4096:8
To disable deterministic mode use the option --not_deterministic
"""

# Not using video reading library from torchvision.
# It only works with old versions of ffmpeg.
import argparse
import csv
import ffmpeg
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

from models.alexnet import AlexLikeNet
from models.bennet import BenNet
from models.resnet import (ResNet18, ResNet34)
from models.resnext import (ResNext34, ResNext50)
from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)

# Need a special comparison function that won't attempt to do something that tensors do not
# support. Used if args.save_top_n or args.save_worst_n are used.
class MaxNode:
    def __init__(self, score, data, metadata, mask):
        self.score = score
        self.data = data
        self.metadata = metadata
        self.mask = mask

    def __lt__(self, other):
        return self.score < other.score

# Turns the heapq from a max heap into a min heap by using greater than in the less than operator.
class MinNode:
    def __init__(self, score, data, metadata, mask):
        self.score = score
        self.data = data
        self.metadata = metadata
        self.mask = mask

    def __lt__(self, other):
        return self.score > other.score

# Argument parser setup for the program.
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
    default=True,
    action='store_false',
    help='Set this to disable deterministic training.')

args = parser.parse_args()

torch.use_deterministic_algorithms(not args.not_deterministic)
torch.manual_seed(args.seed)
random.seed(0)
numpy.random.seed(0)

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
    .shuffle(20000//in_frames, initial=20000//in_frames)
    .decode("l")
    .to_tuple(*decode_strs)
)

batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size)

if args.evaluate:
    eval_dataset = (
        wds.WebDataset(args.evaluate)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=0, batch_size=batch_size)


# Hard coding the Alexnet like network for now.
# TODO Also hard coding the input and output sizes
lr_scheduler = None
train_as_classifier = True
if 'alexnet' == args.modeltype:
    net = AlexLikeNet(in_dimensions=(in_frames, 400, 400), out_classes=3, linear_size=512).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,7], gamma=0.2)
elif 'resnet18' == args.modeltype:
    net = ResNet18(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=True).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-5)
elif 'resnet34' == args.modeltype:
    net = ResNet34(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=True).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
elif 'bennet' == args.modeltype:
    net = BenNet(in_dimensions=(in_frames, 400, 400), out_classes=3).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
elif 'resnext50' == args.modeltype:
    net = ResNext50(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=True).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3], gamma=0.1)
    batch_size = 64
elif 'resnext34' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext34(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=False,
            use_dropout=False).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'resnext18' == args.modeltype:
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext18(in_dimensions=(in_frames, 400, 400), out_classes=3, expanded_linear=True,
            use_dropout=False).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextxt' == args.modeltype:
    net = ConvNextExtraTiny(in_dimensions=(in_frames, 400, 400), out_classes=3).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4, weight_decay=10e-4, momentum=0.9,
            nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5,12], gamma=0.2)
elif 'convnextt' == args.modeltype:
    net = ConvNextTiny(in_dimensions=(in_frames, 400, 400), out_classes=3).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnexts' == args.modeltype:
    net = ConvNextSmall(in_dimensions=(in_frames, 400, 400), out_classes=3).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextb' == args.modeltype:
    net = ConvNextBase(in_dimensions=(in_frames, 400, 400), out_classes=3).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
print(f"Model is {net}")

# See if the model weights and optimizer state should be restored.
if args.resume_from is not None:
    checkpoint = torch.load(args.resume_from)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])
    # Also restore the RNG states
    random.setstate(checkpoint["py_random_state"])
    numpy.random.set_state(checkpoint["np_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

if train_as_classifier:
    #total = 11. + 16. + 12.
    #loss_fn = torch.nn.NLLLoss(weight=torch.tensor([total/11., total/16., total/12.]).cuda())
    loss_fn = torch.nn.NLLLoss()
else:
    #loss_fn = torch.nn.L1Loss()
    loss_fn = torch.nn.MSELoss()

# Gradient scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

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
            # Normalize inputs: input = (input - mean)/stddev
            if args.normalize:
                v, m = torch.var_mean(net_input)
                net_input = (net_input - m) / v

            with torch.cuda.amp.autocast():
                out = net.forward(net_input.contiguous())

                # Convert the labels to a one hot encoding to serve at the DNN target.
                # The label class is 1 based, but need to be 0-based for the one_hot function.
                labels = dl_tuple[-2]
                if train_as_classifier:
                    loss = loss_fn(out, labels.cuda() - 1)
                else:
                    labels = torch.nn.functional.one_hot(labels-1, num_classes=3)
                    loss = loss_fn(out, labels.cuda().float() - 1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                    # Normalize inputs: input = (input - mean)/stddev
                    if args.normalize:
                        v, m = torch.var_mean(net_input)
                        net_input = (net_input - m) / v

                    with torch.cuda.amp.autocast():
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
    print("Evaluating model.")
    if args.save_top_n is not None:
        topn_path = args.outname.split('.')[0] + "-topN"
        # Create the directory if it does not exist
        try:
            os.mkdir(topn_path)
        except FileExistsError:
            pass
        print(f"Saving {args.save_top_n} highest confidence images to {topn_path}.")
        # Save tuples of (class score, tensor)
        topn = [[], [], []]
    if args.save_worst_n is not None:
        worstn_path = args.outname.split('.')[0] + "-worstN"
        # Create the directory if it does not exist
        try:
            os.mkdir(worstn_path)
        except FileExistsError:
            pass
        print(f"Saving {args.save_worst_n} lowest confidence images to {worstn_path}.")
        # Save tuples of (class score, tensor)
        worstn = [[], [], []]

    net.eval()
    with torch.no_grad():
        # Make a confusion matrix, x is item, y is classification
        totals=[[0,0,0],
                [0,0,0],
                [0,0,0]]
        with open(args.outname.split('.')[0] + ".log", 'w') as logfile:
            logfile.write('video_path,frame,time,label,prediction\n')
            for batch_num, dl_tuple in enumerate(eval_dataloader):
                metadata = dl_tuple[-1]
                # Decoding only the luminance channel means that the channel dimension has gone away here.
                if 1 == in_frames:
                    net_input = dl_tuple[0].unsqueeze(1).cuda()
                else:
                    raw_input = []
                    for i in range(in_frames):
                        raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                    net_input = torch.cat(*(raw_input), dim=1)
                # Normalize inputs: input = (input - mean)/stddev
                if args.normalize:
                    v, m = torch.var_mean(net_input)
                    net_input = (net_input - m) / v

                # Visualization masks are not supported with all model types yet.
                if args.modeltype in ['bennet', 'resnet18', 'resnet34']:
                    out, mask = net.vis_forward(net_input)
                else:
                    out = net.forward(net_input)
                    mask = [None] * batch_size
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
                            # If this is a member of the j'th class
                            if 1 == labels[i][j]:
                                totals[j][classes[i]] += 1
                                logfile.write(','.join((metadata[i], str(j), str(classes[i].item()))))
                                logfile.write('\n')
                    if args.save_top_n is not None:
                        for i in range(labels.size(0)):
                            for j in range(3):
                                # If the DNN predicted this image was a member of class j that
                                # should be inserted into the top_n queue.
                                if j == classes[i].item() and 1 == labels[i][j]:
                                    # Insert into an empty heap or replace the smallest value and
                                    # heapify. The smallest value is in the first position.
                                    if len(topn[j]) < args.save_top_n:
                                        heapq.heappush(topn[j], MaxNode(out[i][j].item(), dl_tuple[0][i], metadata[i], mask[i]))
                                    elif out[i][j] > topn[j][0].score:
                                        heapq.heapreplace(topn[j], MaxNode(out[i][j].item(), dl_tuple[0][i], metadata[i], mask[i]))
                    if args.save_worst_n is not None:
                        for i in range(labels.size(0)):
                            for j in range(3):
                                # If the DNN predicted this image was a member of class j that
                                # should be inserted into the worst_n queue.
                                if 1 == labels[i][j]:
                                    # Insert into an empty heap or replace the smallest value and
                                    # heapify. The smallest value is in the first position.
                                    if len(worstn[j]) < args.save_worst_n:
                                        heapq.heappush(worstn[j], MinNode(out[i][j].item(), dl_tuple[0][i], metadata[i], mask[i]))
                                    elif out[i][j] < worstn[j][0].score:
                                        heapq.heapreplace(worstn[j], MinNode(out[i][j].item(), dl_tuple[0][i], metadata[i], mask[i]))

        # The heap has tuples of (class score, tensor)
        if args.save_top_n is not None:
            for j in range(3):
                for i, node in enumerate(topn[j]):
                    img = transforms.ToPILImage()(node.data).convert('L')
                    timestamp = node.metadata.split(',')[2].replace(' ', '_')
                    img.save(f"{topn_path}/class-{j}_time-{timestamp}_score-{node.score}.png")
                    if node.mask is not None:
                        # Save the mask
                        mask_img = transforms.ToPILImage()(node.mask.data).convert('L')
                        mask_img.save(f"{topn_path}/class-{j}_time-{timestamp}_score-{node.score}_mask.png")
        if args.save_worst_n is not None:
            for j in range(3):
                for i, node in enumerate(worstn[j]):
                    img = transforms.ToPILImage()(node.data).convert('L')
                    timestamp = node.metadata.split(',')[2].replace(' ', '_')
                    img.save(f"{worstn_path}/class-{j}_time-{timestamp}_score-{node.score}.png")
                    if node.mask is not None:
                        # Save the mask
                        mask_img = transforms.ToPILImage()(node.mask.data).convert('L')
                        mask_img.save(f"{worstn_path}/class-{j}_time-{timestamp}_score-{node.score}_mask.png")

        # Print evaluation information
        print(f"Evaluation confusion matrix:")
        print(totals)
        accuracy = (totals[0][0] + totals[1][1] + totals[2][2])/(sum(totals[0]) + sum(totals[1]) + sum(totals[2]))
        print(f"Accuracy: {accuracy}")
