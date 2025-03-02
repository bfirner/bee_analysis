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
# ---------------------- Imports ----------------------
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

# Added: Importing torchvision.transforms and logging for later use.
from torchvision import transforms
import logging

from utility.dataset_utility import extractVectors, getImageSize, getVectorSize
from utility.eval_utility import (
    ConfusionMatrix,
    OnlineStatistics,
    RegressionResults,
    WorstExamples,
)
from utility.model_utility import restoreModelAndState
from utility.train_utility import (
    LabelHandler,
    evalEpoch,
    trainEpoch,
    updateWithScaler,
    updateWithoutScaler,
)
from utility.saliency_utils import plot_gradcam_for_multichannel_input

from models.alexnet import AlexLikeNet
from models.bennet import BenNet
from models.resnet import ResNet18, ResNet34
from models.resnext import ResNext18, ResNext34, ResNext50
from models.convnext import ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase
from models.modules import Denormalizer, Normalizer

# ---------------------- Argument Parser ----------------------
# Added: Set up the command-line arguments as per the provided instructions.
parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set."
)
# (Keep existing arguments from first version)
parser.add_argument(
    "--template",
    required=False,
    default=None,
    choices=["bees", "multilabel_detection"],
    type=str,
    help=(
        "Set other options automatically based upon a typical training template."
        "Template settings are overriden by other selected options."
        "bees: Alexnet model with index labels are converted to one hot labels."
        'multilabel: Multilabels are loaded from "detection.pth", binary cross entropy loss is used.'
    ),
)
parser.add_argument("dataset", nargs="+", type=str, help="Dataset for training.")

parser.add_argument(
    "--sample_frames",
    type=int,
    required=False,
    default=1,
    help="Number of frames in each sample.",
)

parser.add_argument(
    "--outname",
    type=str,
    required=False,
    default="model.checkpoint",
    help="Base name for model, checkpoint, and metadata saving.",
)

parser.add_argument(
    "--resume_from", type=str, required=False, help="Model weights to restore."
)

parser.add_argument(
    "--epochs", type=int, required=False, default=15, help="Total epochs to train."
)

parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default="0",
    help="Seed to use for RNG initialization.",
)

parser.add_argument(
    "--normalize",
    required=False,
    default=False,
    action="store_true",
    help="Normalize inputs: input = (input - mean) / stddev.",
)

parser.add_argument(
    "--normalize_outputs",
    required=False,
    default=False,
    action="store_true",
    help="Normalize the outputs. (For regression loss only)",
)

parser.add_argument(
    "--modeltype",
    type=str,
    required=False,
    default="resnext18",
    choices=[
        "alexnet",
        "resnet18",
        "resnet34",
        "bennet",
        "resnext50",
        "resnext34",
        "resnext18",
        "convnextxt",
        "convnextt",
        "convnexts",
        "convnextb",
    ],
    help="Model to use for training.",
)

parser.add_argument(
    "--no_train",
    required=False,
    default=False,
    action="store_true",
    help="Set this flag to skip training. Useful to load an already trained model for evaluation.",
)
parser.add_argument(
    "--evaluate",
    type=str,
    required=False,
    default=None,
    help="Evaluate with given dataset.",
)

parser.add_argument(
    "--save_top_n",
    type=int,
    required=False,
    default=None,
    help="Save N images for class with highest prediction score (with --evaluate).",
)

parser.add_argument(
    "--save_worst_n",
    type=int,
    required=False,
    default=None,
    help="Save N images for class with lowest prediction score (with --evaluate).",
)

parser.add_argument(
    "--not_deterministic",
    required=False,
    default=False,
    action="store_true",
    help="Disable deterministic training.",
)

parser.add_argument(
    "--labels",
    # TODO Support an array of strings to have multiple different label targets.
    type=str,
    nargs="+",
    required=False,
    default=["cls"],
    help="Files to decode from webdataset as the DNN output target labels.",
)

parser.add_argument(
    "--vector_inputs",
    # Support an array of strings to have multiple different label targets.
    type=str,
    nargs="+",
    required=False,
    default=[],
    help="Files to decode from webdataset as DNN vector inputs.",
)

parser.add_argument(
    "--skip_metadata",
    required=False,
    default=False,
    action="store_true",
    help="Skip loading metadata.txt from the webdataset.",
)

parser.add_argument(
    "--convert_idx_to_classes",
    required=False,
    default=1,
    choices=[0, 1],
    type=int,
    help="Convert labels to one-hot if set to 1.",
)

parser.add_argument(
    "--num_outputs",
    required=False,
    default=3,
    type=int,
    help="Number of one-hot elements if converting label indices.",
)

parser.add_argument(
    "--label_offset",
    required=False,
    default=1,
    type=int,
    help='Starting value of classes for "cls" labels.',
)

parser.add_argument(
    "--loss_fun",
    required=False,
    default="CrossEntropyLoss",
    choices=[
        "NLLLoss",
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
        "L1Loss",
        "MSELoss",
        "BCELoss",
    ],
    type=str,
    help="Loss function to use.",
)

# ---------------------- New GradCAM & Debug Options ----------------------
parser.add_argument(
    "--gradcam_cnn_model_layer",
    nargs="+",
    required=False,
    choices=[
        "model_a.0.0",
        "model_a.1.0",
        "model_a.2.0",
        "model_a.3.0",
        "model_a.4.0",
        "model_b.0.0",
        "model_b.1.0",
        "model_b.2.0",
        "model_b.3.0",
        "model_b.4.0",
    ],
    default=["model_a.4.0", "model_b.4.0"],
    help="Model layers for gradcam plots.",
)
parser.add_argument(
    "--debug",
    required=False,
    action="store_true",
    default=False,
    help="Enable debugging output.",
)

args = parser.parse_args()

# ---------------------- Setup Logging and Device ----------------------
# Added: Configure logging and determine the device to use.
logging.basicConfig(
    format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
logging.info(f"Parsed arguments: {args}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ---------------------- Environment Setup ----------------------
# Added: Log system information and set random seeds and deterministic mode.
python_log = os.system("which python3")
machine_log = os.system("uname -a")
date_log = os.system("date")

torch.use_deterministic_algorithms(not args.not_deterministic)
torch.manual_seed(args.seed)
random.seed(0)
numpy.random.seed(0)

if args.template is not None:
    # Elias: the equalities seemed a bit wierd, so I changed it to just use the args
    # and the variable are in the front now
    if args.template == "bees":
        if "--modeltype" not in sys.argv:
            args.modeltype = "alexnet"
    elif args.template == "multilabel_detection":
        if "--modeltype" not in sys.argv:
            args.modeltype = "bennet"
        if "--skip_metadata" not in sys.argv:
            args.skip_metadata = True
        if "--convert_idx_to_classes" not in sys.argv:
            args.convert_idx_to_classes = 0
        if "--labels" not in sys.argv:
            args.labels = ["detection.pth"]
        if "--loss_fun" not in sys.argv:
            args.loss_fun = "BCEWithLogitsLoss"

# ---------------------- Loss Function & Label Preprocessing ----------------------
# Added: Determine the loss function and configure label processing based on settings.
loss_fn = getattr(torch.nn, args.loss_fun)().to(device=device)

regression_loss = ["L1Loss", "MSELoss"]

in_frames = args.sample_frames
decode_strs = []

# Added: Collect decoding strings for image frames.
# The image for a particular frame
for i in range(in_frames):
    decode_strs.append(f"{i}.png")

# Added: Append labels and vector inputs decode strings.
# The class label(s) or regression targets
label_range = slice(len(decode_strs), len(decode_strs) + len(args.labels))
for label_str in args.labels:
    decode_strs.append(label_str)

# Vector inputs (if there are none then the slice will be an empty range)
vector_range = slice(label_range.stop, label_range.stop + len(args.vector_inputs))
for vector_str in args.vector_inputs:
    decode_strs.append(vector_str)

# Metadata for this sample. A string of format: f"{video_path},{frame},{time}"
# Added: Append metadata if not skipped.
if not args.skip_metadata:
    metadata_index = len(decode_strs)
    decode_strs.append("metadata.txt")


if args.labels[0] != "cls":
    label_offset = 0
else:
    label_offset = args.label_offset
logging.info(f"Adjusting labels with offset {label_offset}")

# If we are converting to a one-hot encoding output then we need to check the argument that
# specifies the number of output elements. Otherwise we can check the number of elements in the
# webdataset.
if args.convert_idx_to_classes == 1:
    label_size = (
        getVectorSize(args.dataset, decode_strs, label_range) * args.num_outputs
    )
else:
    label_size = getVectorSize(args.dataset, decode_strs, label_range)

# See if we can deduce the label names
label_names = None
if args.convert_idx_to_classes != 1:
    label_names = []
    for label_idx in range(len(args.labels)):
        label_names.append(args.labels[label_idx])

label_handler = LabelHandler(label_size, label_range, label_names)

# The label value may need to be adjusted, for example if the label class is 1 based, but
# should be 0-based for the one_hot function. This is done by subtracting the label_offset from the
# labels.
# Also scale the labels before calculating loss to rebalance how loss is distributed across the
# labels and to put the labels in a better training range. Note that this only makes sense with a
# regression loss, where the label_offset adjustment would not be used.
if args.normalize_outputs:
    logging.info("Reading dataset to compute label statistics for normalization.")
    label_stats = [OnlineStatistics() for _ in range(label_size)]
    label_dataset = wds.WebDataset(args.dataset).to_tuple(*args.labels)
    label_dataloader = torch.utils.data.DataLoader(
        label_dataset, num_workers=0, batch_size=1
    )
    for data in label_dataloader:
        for label, stat in zip(
            extractVectors(data, slice(0, label_size))[0].tolist(), label_stats
        ):
            stat.sample(label)
    label_means = torch.tensor([stat.mean() for stat in label_stats]).cuda()
    label_stddevs = torch.tensor(
        [math.sqrt(stat.variance()) for stat in label_stats]
    ).cuda()
    if (label_stddevs.abs() < 0.0001).any():
        logging.error("Some labels have extremely low variance -- check your dataset.")
        exit(1)
    denormalizer = Denormalizer(means=label_means, stddevs=label_stddevs).to(device)
    normalizer = Normalizer(means=label_means, stddevs=label_stddevs).to(device)
    label_handler.setPreprocess(lambda labels: normalizer(labels))
else:
    denormalizer = None
    normalizer = None
    label_handler.setPreprocess(lambda labels: labels - label_offset)
if args.convert_idx_to_classes == 1:
    label_handler.setPreeval(
        lambda labels: torch.nn.functional.one_hot(
            (labels - label_offset), num_classes=label_handler.size()
        )
    )


# Network outputs may need to be postprocessed for evaluation if some postprocessing is being done
# automatically by the loss function.
if args.loss_fun == "CrossEntropyLoss":
    # Outputs of most classification networks are considered probabilities (but only take that in a
    # very loose sense of the word). The Softmax function forces its inputs to sum to 1 as
    # probabilities should do.
    nn_postprocess = torch.nn.Softmax(dim=1)
elif args.loss_fun == "BCEWithLogitsLoss":
    nn_postprocess = torch.nn.Sigmoid()
else:
    # Otherwise just use an identify function unless normalization is being used.
    nn_postprocess = (
        (lambda x: denormalizer(x)) if denormalizer is not None else (lambda x: x)
    )

# ---------------------- Dataset Setup ----------------------
# Added: Build the webdataset with the given transformations.
# TODO FIXME Deterministic shuffle only shuffles within a range. Should perhaps manipulate what is
# in the tar file by shuffling filenames after the dataset is created.
logging.info(f"Training with dataset {args.dataset}")
dataset = (
    wds.WebDataset(args.dataset, shardshuffle=True)
    .shuffle(20000 // in_frames, initial=20000 // in_frames)
    # TODO This will hardcode all images to single channel numpy float images, but that isn't clear
    # from any documentation.
    # TODO Why "l" rather than decode to torch directly with "torchl"?
    .decode("l")
    .to_tuple(*decode_strs)
)
image_size = getImageSize(args.dataset, decode_strs)
logging.info(f"Decoding images of size {image_size}")

batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size)
if args.evaluate:
    eval_dataset = wds.WebDataset(args.evaluate).decode("l").to_tuple(*decode_strs)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, num_workers=0, batch_size=batch_size
    )
    logging.info(f"Loaded evaluation dataset from {args.evaluate}")

# ---------------------- Model Setup ----------------------
# Added: Configure model arguments and instantiate the chosen model.
model_args = {
    "in_dimensions": (in_frames, image_size[1], image_size[2]),
    "out_classes": label_handler.size(),
}
vector_input_size = 0
if len(args.vector_inputs) > 0:
    vector_input_size = getVectorSize(args.dataset, decode_strs, vector_range)
    model_args["vector_input_size"] = vector_input_size

skip_last_relu = args.loss_fun in regression_loss
use_amp = False
if args.modeltype == "alexnet":
    model_args["linear_size"] = 512
    model_args["skip_last_relu"] = skip_last_relu
    net = AlexLikeNet(**model_args).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 5, 7], gamma=0.2
    )
    use_amp = True
elif args.modeltype == "resnet18":
    model_args["expanded_linear"] = True
    net = ResNet18(**model_args).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
elif args.modeltype == "resnet34":
    model_args["expanded_linear"] = True
    net = ResNet34(**model_args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
elif args.modeltype == "bennet":
    net = BenNet(**model_args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
elif args.modeltype == "resnext50":
    model_args["expanded_linear"] = True
    net = ResNext50(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1, 2, 3], gamma=0.1
    )
    batch_size = 64
elif args.modeltype == "resnext34":
    model_args["expanded_linear"] = False
    model_args["use_dropout"] = False
    net = ResNext34(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 9], gamma=0.2
    )
elif args.modeltype == "resnext18":
    model_args["expanded_linear"] = True
    model_args["use_dropout"] = False
    net = ResNext18(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 12], gamma=0.2
    )
elif args.modeltype == "convnextxt":
    net = ConvNextExtraTiny(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9, nesterov=True
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4, 5, 12], gamma=0.2
    )
    use_amp = True
elif args.modeltype == "convnextt":
    net = ConvNextTiny(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 12], gamma=0.2
    )
elif args.modeltype == "convnexts":
    net = ConvNextSmall(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-2, weight_decay=1e-4, momentum=1e-3
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 12], gamma=0.2
    )
elif args.modeltype == "convnextb":
    net = ConvNextBase(**model_args).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 12], gamma=0.2
    )
logging.info(f"Model is {net}")

if args.resume_from is not None:
    restoreModelAndState(args.resume_from, net, optimizer)

# ---------------------- Training Loop ----------------------
# Added: Wrap the training loop in a try/except to log and re-raise exceptions.
if not args.no_train:
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    try:
        worst_training = None
        if args.save_worst_n is not None:
            worst_training = WorstExamples(
                args.outname.split(".")[0] + "-worstN-train",
                (
                    label_names
                    if label_names is not None
                    else [str(i) for i in range(label_size)]
                ),
                args.save_worst_n,
            )
            logging.info(
                f"Saving worst training examples to {worst_training.worstn_path}."
            )
        for epoch in range(args.epochs):
            logging.info(f"Starting epoch {epoch}")
            if args.loss_fun in regression_loss:
                totals = RegressionResults(
                    size=label_handler.size(), names=label_handler.names()
                )
            else:
                totals = ConfusionMatrix(size=label_handler.size())

            trainEpoch(
                net=net,
                optimizer=optimizer,
                scaler=scaler,
                label_handler=label_handler,
                train_stats=totals,
                dataloader=dataloader,
                vector_range=vector_range,
                train_frames=in_frames,
                normalize_images=args.normalize,
                loss_fn=loss_fn,
                nn_postprocess=nn_postprocess,
                worst_training=worst_training,
                skip_metadata=args.skip_metadata,
            )
            # Adjust learning rate according to the learning rate schedule
            if lr_scheduler is not None:
                lr_scheduler.step()
            # Save checkpoint
            torch.save(
                {
                    "model_dict": net.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "py_random_state": random.getstate(),
                    "np_random_state": numpy.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "denormalizer_state_dict": (
                        denormalizer.state_dict() if denormalizer is not None else None
                    ),
                    "normalizer_state_dict": (
                        normalizer.state_dict() if normalizer is not None else None
                    ),
                    "metadata": {
                        "modeltype": args.modeltype,
                        "labels": args.labels,
                        "vector_inputs": args.vector_inputs,
                        "convert_idx_to_classes": args.convert_idx_to_classes,
                        "label_size": label_handler.size(),
                        "model_args": model_args,
                        "normalize_images": args.normalize,
                        "normalize_labels": args.normalize_outputs,
                    },
                },
                args.outname,
            )
            # Evaluation step during training if requested
            # Validation step if requested
            if args.evaluate is not None:
                print(f"Evaluating epoch {epoch}")
                if args.loss_fun in regression_loss:
                    eval_totals = RegressionResults(
                        size=label_handler.size(), names=label_handler.names()
                    )
                else:
                    eval_totals = ConfusionMatrix(size=label_handler.size())
                evalEpoch(
                    net=net,
                    label_handler=label_handler,
                    eval_stats=eval_totals,
                    eval_dataloader=eval_dataloader,
                    # ! What is eval range?
                    vector_range=eval_range,
                    train_frames=in_frames,
                    normalize_images=args.normalize,
                    loss_fn=loss_fn,
                    nn_postprocess=nn_postprocess,
                )
            # End training loop; final checkpoint saved above.
    except Exception as e:
        logging.error(f"Exception during training: {e}")
        raise e
    

if args.loss_fun not in regression_loss:
    if 'CrossEntropyLoss' == args.loss_fun:
        # Outputs of most classification networks are considered probabilities (but only take that in a
        # very loose sense of the word). The Softmax function forces its inputs to sum to 1 as
        # probabilities should do.
        sm = torch.nn.Softmax(dim=1)
        nn_postprocess = lambda classes: torch.round(sm(classes)).clamp(0, 1)
    elif 'BCEWithLogitsLoss' == args.loss_fun:
        sigm = torch.nn.Sigmoid()
        nn_postprocess = lambda classes: torch.round(sigm(classes)).clamp(0, 1)
    else:
        nn_postprocess = lambda classes: torch.round(classes).clamp(0, 1)


# ---------------------- Post-Training Evaluation & GradCAM ----------------------
# Added: If evaluation dataset was provided, perform post-training evaluation and optionally generate GradCAM plots.
if args.evaluate:
    logging.info("Starting post-training evaluation.")
    top_eval = None
    worst_eval = None
    if args.save_top_n is not None:
        top_eval = WorstExamples(
            args.outname.split(".")[0] + "-topN-eval",
            (
                label_names
                if label_names is not None
                else [str(i) for i in range(label_size)]
            ),
            args.save_top_n,
            worst_mode=False,
        )
        logging.info(f"Saving top evaluation examples to {top_eval.worstn_path}.")
    if args.save_worst_n is not None:
        worst_eval = WorstExamples(
            args.outname.split(".")[0] + "-worstN-eval",
            (
                label_names
                if label_names is not None
                else [str(i) for i in range(label_size)]
            ),
            args.save_worst_n,
        )
        logging.info(f"Saving worst evaluation examples to {worst_eval.worstn_path}.")
    net.eval()
    with torch.no_grad():
        # Make a confusion matrix or loss statistics
        if args.loss_fun in regression_loss:
            totals = RegressionResults(size=label_size)
        else:
            totals = ConfusionMatrix(size=label_size)

        with open(args.outname.split(".")[0] + ".log", "w") as logfile:
            logfile.write("video_path,frame,time,label,prediction\n")
            for batch_num, dl_tuple in enumerate(eval_dataloader):
                # Decoding only the luminance channel means that the channel dimension has gone away here.
                if in_frames == 1:
                    net_input = dl_tuple[0].unsqueeze(1).to(device=device)
                else:
                    raw_input = [
                        dl_tuple[i].unsqueeze(1).to(device=device)
                        for i in range(in_frames)
                    ]
                    net_input = torch.cat(raw_input, dim=1)
                # Normalize inputs: input = (input - mean)/stddev
                if args.normalize:
                    # Normalize per channel, so compute over height and width
                    v, m = torch.var_mean(net_input, dim=(-2, -1), keepdim=True)
                    net_input = (net_input - m) / v
                
                # GradCAM plotting if enabled (only for alexnet type with gradcam layers)
                if args.gradcam_cnn_model_layer and args.modeltype in [
                    "alexnet",
                    "bennet",
                    "resnet18",
                    "resnet34",
                ]:
                    target_classes = (
                        extractVectors(dl_tuple, label_range).cpu().tolist()
                    )
                    model_names = ["model_a", "model_b"]
                    for last_layer, model_name in zip(
                        args.gradcam_cnn_model_layer, model_names
                    ):
                        try:
                            num_cls = (
                                len(set(target_classes)) if set(target_classes) else 3
                            )
                            logging.info(f"Plotting GradCAM for layer {last_layer}")
                            plot_gradcam_for_multichannel_input(
                                model=net,
                                dataset=os.path.basename(args.evaluate).split(".")[0],
                                input_tensor=net_input,
                                target_layer_name=[last_layer],
                                model_name=model_name,
                                target_classes=target_classes,
                                number_of_classes=num_cls,
                            )
                        except Exception as e:
                            logging.error(f"GradCAM error for layer {last_layer}: {e}")
 
                vector_input=None
                if 0 < vector_input_size:
                    vector_input = extractVectors(dl_tuple, vector_range).cuda()

                # Visualization masks are not supported with all model types yet.
                if args.modeltype in ['alexnet', 'bennet', 'resnet18', 'resnet34']:
                    out, mask = net.vis_forward(net_input, vector_input)
                else:
                    out = net.forward(net_input, vector_input, vector_input)
                    mask = [None] * batch_size

                # Convert the labels to a one hot encoding to serve at the DNN target.
                # The label class is 1 based, but need to be 0-based for the one_hot function.
                labels = extractVectors(dl_tuple,label_range).cuda()

                if args.skip_metadata:
                    metadata = [""] * labels.size(0)
                else:
                    metadata = dl_tuple[metadata_index]

                loss = loss_fn(out, label_handler.preprocess(labels))

                # Fill in the loss statistics and best/worst examples
                with torch.no_grad():
                    # The postprocessesing could include Softmax, denormalization, etc.
                    post_out = nn_postprocess(out)
                    # Labels may also require postprocessing, for example to convert to a one-hot
                    # encoding.
                    post_labels = label_handler.preeval(labels)

                    # Log the predictions
                    for i in range(post_labels.size(0)):
                        logfile.write(','.join((metadata[i], str(out[i]), str(post_labels[i]))))
                        logfile.write('\n')
                    if worst_eval is not None or top_eval is not None:
                        # For each item in the batch see if it requires an update to the worst examples
                        # If the DNN should have predicted this image was a member of the labelled class
                        # then see if this image should be inserted into the worst_n queue for the
                        # labelled class based upon the DNN output for this class.
                        input_images = dl_tuple[0]
                        for i in range(post_labels.size(0)):
                            label = torch.argwhere(post_labels[i])[0].item()
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