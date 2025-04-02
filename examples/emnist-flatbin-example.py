#!/usr/bin/python3
"""
Copyright © 2025 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

This file demonstrates ML repository. Conversion to flatbin data and training are demonstrated.
"""

import argparse
import importlib
import numpy
import os
import random
import sys
import time
import torch
import torchvision
from torchvision.transforms import v2 as transforms

# Insert the ml repository into the path so that the python modules are used properly
# This hack allows a user to run the script from the top level directory or from the examples directory.
sys.path.append('./')
sys.path.append('../')

import models.fleximodel as fleximodel
import utility.flatbin_dataset as flatbin_dataset
import utility.train_utility as train_utility

from emnist_common import (NormalizeImages, MakeLabelsOneHot, get_torch_dataset, flatbin_path, get_example_datum, get_dataset_classes, remake_dataset)

parser = argparse.ArgumentParser(description='Demonstration of different approaches to classification.')

parser.add_argument(
    '--seed',
    type=int,
    required=False,
    default=0,
    help='Seed for torch.random.manual_seed')

parser.add_argument(
    '--use_amp',
    required=False,
    default=False,
    action="store_true",
    help='Use automatic mixed precision loss')

parser.add_argument(
    '--train_device',
    type=str,
    required=False,
    default=None,
    help="Which device to use for training (typically 'cuda' or 'cpu'). Set automatically if not provided.")

parser.add_argument(
    '--test',
    type=str,
    required=True,
    choices=['torch_dataloader', 'flatbin_dataloader'],
    help='Which test to perform')

parser.add_argument(
    '--dataroot',
    type=str,
    required=True,
    help="Path to find and store emnist data.")

parser.add_argument(
    '--epochs',
    type=int,
    required=False,
    default=50,
    help="Number of epochs to train.")

args = parser.parse_args()

################
# Training set up

# Set the seed as specified
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)

# Determine the training device
if args.train_device is None:
    if torch.backends.cuda.is_built():
        train_device = 'cuda'
    else:
        train_device = 'cpu'
else:
    train_device = args.train_device

# Gradient scaler for mixed precision training
if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

################
# Testing

if args.test == "torch_dataloader":
    torch.random.manual_seed(args.seed)

    # We need to find some details about the dataset
    preprocess = transforms.Compose([
        #transforms.ToImageTensor(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
    ])

    probe_dataset = get_torch_dataset(args.dataroot, train=True, transform=preprocess)

    # Figure out some dataset information
    example_img, example_label = get_example_datum(probe_dataset)
    max_class = get_dataset_classes(probe_dataset)
    print(f"Image size is {example_img[0].size()} and there are {max_class} classes.")
    hyperparams = fleximodel.make_small_hyperparams(in_channels=example_img.size(0), num_outputs=max_class)

    # Remake the dataset and include all preprocessing steps this time
    preprocess = transforms.Compose([
        #transforms.ToImageTensor(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
        # Normalizing (0 mean, unit variance) isn't strictly necessary, depending upon the dataset, but it generally improves training.
        # Always use CPU because the dataloader is running in forked processes
        NormalizeImages(torch.device('cpu')),
        #MakeLabelsOneHot(max_class, torch.device(train_device))
    ])

    train_dataset = get_torch_dataset(args.dataroot, train=True, transform=preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=3, drop_last=False, pin_memory=True)

    eval_dataset = get_torch_dataset(args.dataroot, train=True, transform=preprocess)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=3, drop_last=False, pin_memory=True)

    # Notes:
    # On my computer the times per batch are:
    # CPU with the emnist dataloader, 3.0-3.5ms/batch
    # GPU with the emnist dataloader, 2.3-2.5ms/batch

if args.test == "flatbin_dataloader":
    torch.random.manual_seed(args.seed)

    train_path = flatbin_path(args.dataroot, train=True, split='balanced')
    # Check if the flatbin has been created yet
    if not os.path.exists(train_path):
        remake_dataset(is_training=True)

    eval_path = flatbin_path(args.dataroot, train=False, split='balanced')
    if not os.path.exists(eval_path):
        remake_dataset(is_training=False)

    print("probing")
    probe_dataset = flatbin_dataset.FlatbinDataset(train_path, ["image.png", "class.int"])
    print(f"Data sizes are {probe_dataset.getDataSize(0)} and {probe_dataset.getDataSize(1)}")

    # Figure out some dataset information
    # Note that we can use the readFirst function available from the FlatbinDataset, but we still need to find the number of classes:
    example_img, example_label = probe_dataset.readFirst()
    max_class = get_dataset_classes(probe_dataset)
    print(f"Image size is {example_img.size()} and there are {max_class} classes.")

    # The dataloader for training
    train_dataset = flatbin_dataset.FlatbinDataset(train_path, ["image.png", "class.int"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=3, batch_size=32, pin_memory=True, drop_last=False, persistent_workers=False)

    # The dataloader for evaluation
    eval_dataset = flatbin_dataset.FlatbinDataset(eval_path, ["image.png", "class.int"])
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=3, batch_size=32, pin_memory=True, drop_last=False, persistent_workers=False)

################
# Training

# Set up the model. The example_img and max_class variables should have been set up previously.
hyperparams = fleximodel.make_small_hyperparams(in_channels=example_img.size(0), num_outputs=max_class)
# Typical neural network with lots of parameters.
net = fleximodel.FlexiNet(in_dimensions=example_img.size(), out_classes = max_class, hyperparams=hyperparams)

net.to(train_device)
net.train()

#optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.001, weight_decay=0.00)
optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01, weight_decay=0.001)

print("training")
# The loss should be derived from nn.LogSoftMax and the nn.NLLLoss functions, or with the equivalent nn.CrossEntropyLoss
loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.BCEWithLogitsLoss()

# Notes:
# On my computer the times per batch are:
# CPU with the default dataloader, ms/batch
# GPU with the default dataloader, 2.8-3.2ms ms/batch

for epoch in range(args.epochs):
    print(f"Starting epoch {epoch}")
    begin_time = time.time_ns()
    batch_loss = 0
    for batch_num, dl_tuple in enumerate(train_dataloader):
        images, labels = dl_tuple
        # Images have already been normalized and the flatbin image loader converts pixels into the 0 to 1 range
        images = images.to(train_device)
        #labels = torch.nn.functional.one_hot(labels, num_classes=max_class).float().to(train_device)
        # One hot labels are not required with the CrossEntropyLoss criterion
        labels = labels.to(train_device)

        if scaler is not None:
            out, loss = train_utility.updateWithScaler(loss_fn, net, images, None, labels, scaler, optimizer)
        else:
            out, loss = train_utility.updateWithoutScaler(loss_fn, net, images, None, labels, optimizer)
        batch_loss += loss
    end_time = time.time_ns()
    duration = end_time - begin_time
    print(f"Average batch loss {batch_loss / (batch_num+1)}")
    print(f"Epoch time {duration/10**9} seconds ({round(duration/10**6/(batch_num+1), ndigits=3)}ms / batch)")

################
# Evaluation

net.eval()

print(f"Starting evaluation.")
begin_time = time.time_ns()
batch_loss = 0
total_correct = 0
total_examples = 0
for batch_num, dl_tuple in enumerate(eval_dataloader):
    images, labels = dl_tuple
    # Images have already been normalized and the flatbin image loader converts pixels into the 0 to 1 range
    images = images.to(train_device)
    #labels = torch.nn.functional.one_hot(labels, num_classes=max_class).float().to(train_device)
    # One hot labels are not required with the CrossEntropyLoss criterion
    labels = labels.to(train_device)

    output = net.forward(images)
    batch_loss += loss_fn(output, labels)

    # Get the success rate.
    # Convert network outputs to range [0,1] with softmax and choose anything > 0.5 as the prediction.
    one_hot_predictions = (torch.nn.functional.softmax(output, dim=-1) > 0.5).long()
    # Convert those predictions to class labels
    predictions = torch.argmax(one_hot_predictions, dim = -1)
    # Labels are the class numbers, so just check for equality
    total_correct += torch.sum(predictions == labels).item()
    total_examples += labels.size(0)


end_time = time.time_ns()
duration = end_time - begin_time
print(f"Average evaluation batch loss {batch_loss / (batch_num+1)}")
print(f"Evaluation time {duration/10**9} seconds ({round(duration/10**6/(batch_num+1), ndigits=3)}ms / batch)")
print(f"Correct predictions {total_correct}/{total_examples}, {round(100*total_correct/total_examples, ndigits=4)}%")

