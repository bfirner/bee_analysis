"""
Module Name: make_validation_training.py

Description:
    Sets up k-fold cross‑validation for video behavior discriminator pipelines on Rutgers CS infrastructure.
    - Splits a master dataset CSV into k balanced folds.
    - Generates per‑fold CSVs (dataset_0.csv … dataset_{k-1}.csv).
    - Creates N data preparation batch scripts (VidActRecDataprep.py) for each fold.
    - Builds a global sbatch script to submit all data prep jobs.
    - Generates N training scripts (VidActRecTrain.py) for each fold.
    - Builds a training sbatch submission script using srun for GPU jobs.

Usage:
    python make_validation_training.py \
        --in-path <input_dir> \
        --out-path <output_dir> \
        --datacsv <dataset.csv> \
        --k <num_folds> \
        [--batchdir <batch_dir>] \
        [--seed <seed>] \
        [--training <training_base>] \
        [--model <model_name>] \
        [--only_split] \
        [--width <px>] [--height <px>] \
        [--crop_x_offset <px>] [--crop_y_offset <px>] \
        [--label_offset <offset>] \
        [--training_only] \
        [--frames_per_sample <n>] \
        [--epochs <n>] \
        [--gpus <n>] \
        [--remove-dataset-sub] \
        [--gradcam_cnn_model_layer <layers>...] \
        [--time-to-run-training <seconds>] \
        [--num-outputs <n>] \
        [--path_to_file <bee_analysis_dir>] \
        [--binary-training-optimization] \
        [--use-dataloader-workers] \
        [--max-dataloader-workers <n>] \
        [--loss-fn <loss_function>]

Arguments:
    --in-path                   Input directory for the master CSV. (default: “.”)
    --out-path                  Output directory for generated CSVs and scripts. (default: “.”)
    --datacsv                   Name of the master dataset CSV. (default: “dataset.csv”)
    --k                         Number of folds to create. (default: 3)
    --batchdir                  Working directory for sbatch jobs. (default: “.”)
    --seed                      Random seed for fold shuffling. (default: 01011970)
    --training                  Base name for the main training batch script. (default: “training-run”)
    --model                     Model name for training scripts. (default: “alexnet”)
    --only_split                Only perform CSV splitting; skip script generation.
    --width, --height           Output crop dimensions in pixels. (default: 400×400)
    --crop_x_offset, --crop_y_offset  
                                Crop offsets in pixels. (default: 0,0)
    --label_offset              Class label offset for training. (default: 0)
    --training_only             Generate only training set files. (default: False)
    --frames_per_sample         Frames per sample in training. (default: 1)
    --epochs                    Number of training epochs. (default: 10)
    --gpus                      GPUs per SLURM job. (default: 1)
    --remove-dataset-sub        Skip creating per‑fold dataset CSV files.
    --gradcam_cnn_model_layer   Layers for GradCAM plots. (default: model_a.4.0 model_b.4.0)
    --time-to-run-training      SLURM time limit per job in seconds. (default: 28800)
    --num-outputs               Number of output classes. (default: 3)
    --path_to_file              Directory of VidActRecDataprep.py and VidActRecTrain.py (default: bee_analysis)
    --binary-training-optimization
                                Enable binary training optimization. (default: False)
    --use-dataloader-workers    Use DataLoader workers in training script. (default: False)
    --max-dataloader-workers    Number of DataLoader workers. (default: 3)
    --loss-fn                   Loss function for training. (choices: CrossEntropyLoss, NLLLoss, etc.; default: CrossEntropyLoss)

Workflow:
    1. Read and parse the master dataset CSV; group entries by class.
    2. Shuffle each class’s rows using the provided seed.
    3. Evenly distribute rows into k folds.
    4. Write per‑fold CSV files (`dataset_0.csv` … `dataset_{k-1}.csv`).
       If `--only_split` is set, exit here.
    5. Generate:
         - `training-run.sh`: sets up virtualenv and installs requirements.
         - `train_{i}.sh`: runs VidActRecTrain.py on fold i’s tar/bin files.
         - A master sbatch submission script invoking all training jobs with SLURM directives.
    6. Make all `.sh` and `.log` files executable.

Dependencies:
    - Python 3.x
    - pandas
    - OpenCV (for preprocessing/training scripts)
    - SLURM environment (`sbatch`, `srun`)
"""

import argparse
import csv
import logging
import os
import random
import sys

logging.basicConfig(
    format="%(asctime)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
parser = argparse.ArgumentParser(description="Create k-fold validation sets.")

parser.add_argument(
    "--in-path",
    type=str,
    required=False, 
    default=".",
    help="the input path"
)

parser.add_argument(
    "--out-path",
    type=str,
    required=False,
    default=".",
    help="output file path"
)


parser.add_argument(
    "--datacsv",
    type=str,
    required=False,
    default="dataset.csv",
    help="name of the dataset, default dataset.csv",
)
parser.add_argument(
    "--k", type=int, required=False, default=3, help="number of sets, default 3"
)
parser.add_argument(
    "--batchdir",
    type=str,
    required=False,
    default=".",
    help="working directory for the sbatch run jobs, default: .",
)
parser.add_argument(
    "--venv-path",
    type=str,
    default=".",
    help="the path to the virtual environment that you want to use (absolute probably); default: '.'"
)
parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default="01011970",
    help="Seed to use for randominizing the data sets, default: 01011970",
)
parser.add_argument(
    "--training",
    type=str,
    required=False,
    default="training-run",
    help="Name for the training script file, default: training-run",
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    default="alexnet",
    help="Model to use for the training script, default: alexnet",
)
parser.add_argument(
    "--only_split",
    required=False,
    default=False,
    action="store_true",
    help="Set to finish after splitting the csv, default: False",
)
parser.add_argument(
    "--width",
    type=int,
    required=False,
    default=400,
    help="Width of output images (obtained via cropping, after applying scale), default 400",
)
parser.add_argument(
    "--height",
    type=int,
    required=False,
    default=400,
    help="Height of output images (obtained via cropping, after applying scale), default 400",
)
parser.add_argument(
    "--crop_x_offset",
    type=int,
    required=False,
    default=0,
    help="The offset (in pixels) of the crop location on the original image in the x dimension, default 0",
)
parser.add_argument(
    "--crop_y_offset",
    type=int,
    required=False,
    default=0,
    help="The offset (in pixels) of the crop location on the original image in the y dimension, default 0",
)
parser.add_argument(
    "--label_offset",
    required=False,
    default=0,
    type=int,
    help='The starting value of classes when training with cls labels (the labels value is "cls"), default: 0',
)
parser.add_argument(
    "--training_only",
    type=bool,
    required=False,
    default=False,
    help="only generate the training set files, default: False",
)
parser.add_argument(
    "--frames_per_sample",
    type=int,
    required=False,
    default=1,
    help="Number of frames per sample, default 1",
)
parser.add_argument(
    "--epochs",
    type=int,
    required=False,
    default=10,
    help="Number of epochs to run, default 10",
)
parser.add_argument(
    "--gpus",
    required=False,
    default=1,
    type=int,
    help="Number of GPUs to use, default 1",
)
parser.add_argument(
    "--remove-dataset-sub",
    required=False,
    default=False,
    action="store_true",
    help="don't create the dataset_*.csv files",
)
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
    "--time-to-run-training",
    required=False,
    help="Time limit to run the training jobs, default 28800 minutes (20 days)",
    type=int,
    default=28800,
)
parser.add_argument(
    "--num-outputs",
    required=False,
    help="the number of outputs/classes that are required, used for the train command",
    default=3,
    type=int,
)
parser.add_argument(
    "--path_to_file",
    type=str,
    required=False,
    default="bee_analysis",
    help="path to bee analysis files, default: bee_analysis",
)
# flatbin stuff
parser.add_argument(
    "--binary-training-optimization",
    action="store_true",
    required=False,
    help="Convert and train with binary files",
    default=False,
)

parser.add_argument(
    "--use-dataloader-workers",
    action="store_true",
    default=False,
    required=False,
    help="Whether to use dataloader workers in the training script.",
)

parser.add_argument(
    "--max-dataloader-workers",
    type=int,
    default=3,
    required=False,
    help="The number of dataloader workers, default=3. Only works when the `--use-dataloader-workers` flag is passed.",
)

parser.add_argument(
    "--loss-fn",
    type=str,
    default="CrossEntropyLoss",
    choices=[
        "NLLLoss",
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
        "L1Loss",
        "MSELoss",
        "BCELoss",
    ],
    required=False,
    help="The loss function to be used for the training script",
)

args = parser.parse_args()

# program_dir = "/research/projects/grail/rmartin/analysis-results/code/bee_analysis"
program_dir = os.path.join(os.getcwd(), args.path_to_file)
dataPrepProgram = os.path.join(program_dir, "VidActRecDataprep.py")
# The training python program
trainProgram = os.path.join(program_dir, "VidActRecTrain.py")

datacsvname = args.datacsv
numOfSets = args.k
batchdir = os.path.join(args.out_path, args.batchdir)
seed = args.seed
training_filename = args.training + ".sh"
model_name = args.model
width = args.width
height = args.height
crop_x_offset = args.crop_x_offset
crop_y_offset = args.crop_y_offset
label_offset = args.label_offset
training_only = args.training_only

# command to run the evaluation and training program
# <eval-set> <a-set> <b-set> ...
trainCommand = (
    f"python3 {trainProgram} --num_outputs {args.num_outputs}"
    f" --sample_frames {args.frames_per_sample} "
    f" --gradcam_cnn_model_layer {' '.join(args.gradcam_cnn_model_layer)} "
    f" --not_deterministic --epochs {args.epochs}"
    f" --modeltype {model_name} "
    f" --label_offset {label_offset} "
    f" --loss_fun {args.loss_fn} "
)

if args.binary_training_optimization:
    trainCommand += " --labels cls " " --convert_idx_to_classes 1 " " --skip_metadata "

if args.use_dataloader_workers:
    trainCommand += f" --num_workers {args.max_dataloader_workers} "

# evaluation has to be last because it has to be placed adjacent to the tar files
trainCommand += " --evaluate "

logging.info(f"dataset is {datacsvname}")

# set the random number generator
random.seed(seed)

if not args.remove_dataset_sub:
    with open(os.path.join(args.out_path, datacsvname)) as datacsv:
        conf_reader = csv.reader(datacsv)
        header = next(conf_reader)
        # Remove all spaces from the header strings
        header = ["".join(col.split(" ")) for col in header]
        logging.info(f"header is {header}")
        file_col = header.index("filename")
        class_col = header.index("class")
        beginf_col = header.index("beginframe")
        endf_col = header.index("endframe")

        # Group rows by their class value
        class_groups = {}
        for row in conf_reader:
            cls = row[class_col]
            class_groups.setdefault(cls, []).append(row)

    # Initialize folds (one per dataset file)
    folds = [[] for _ in range(numOfSets)]

    # For each class, shuffle its rows and distribute them evenly across folds
    for cls, rows in class_groups.items():
        random.shuffle(rows)
        for i, row in enumerate(rows):
            fold_index = i % numOfSets
            folds[fold_index].append(row)

    numRows = sum(len(fold) for fold in folds)
    logging.info(
        f"Splitting {numRows} rows into {numOfSets} datasets with balanced classes"
    )

# foreach dataset, construct a csv of the files in that set
baseNameFile = datacsvname.split(".csv")
baseName = baseNameFile[0]
setNum = 0

# Write out the split csv files.
if not args.remove_dataset_sub:
    for dataset_num in range(numOfSets):
        dataset_filename = baseName + "_" + str(dataset_num) + ".csv"
        with open(os.path.join(args.out_path, dataset_filename), "w") as dsetFile:
            # write out the header row at the top of the set
            dsetFile.write("file, class, begin frame, end frame\n")
            # write out all the rows for this set from the corresponding fold
            for row in folds[dataset_num]:
                dsetFile.write(",".join(row))
                dsetFile.write("\n")

# Finish here if the only_split option was set.
if args.only_split:
    sys.exit(0)


training_batch_file = open(os.path.join(args.out_path, training_filename), "w")
training_batch_file.write("#!/usr/bin/bash \n")
training_batch_file.write(f"source {os.path.join(args.venv_path, 'venv/bin/activate')} \n")

training_batch_file.write(f"pip install --no-compile -r {os.path.join(program_dir, 'requirements.txt')}\n")
training_batch_file.write("# batch file for getting the training results \n \n")
training_batch_file.write(
    "echo start-is: `date` \n \n"
)  # add start timestamp to training file

for dataset_num in range(numOfSets):
    train_job_filename = "train" + "_" + str(dataset_num) + ".sh"

    # open the batch file that runs the testing and training commands
    with open(os.path.join(args.out_path, train_job_filename), "w") as trainFile:
        trainFile.write("#!/usr/bin/bash \n")
        
        
        trainFile.write(f"source {os.path.join(args.venv_path, 'venv/bin/activate')} \n")
        trainFile.write("# command to run \n \n")
        trainFile.write("export TRAINPROGRAM=" + trainProgram + "\n")
        trainFile.write("echo start-is: `date` \n \n")  # add start timestamp
        traincommand_local = trainCommand
        traincommand_local = (
            traincommand_local
            + " "
            + f"{baseName}_{str(dataset_num)}.{'tar' if not args.binary_training_optimization else 'bin'}"
        )
        for trainingSetNum in range(numOfSets):
            if int(trainingSetNum) != int(dataset_num):
                traincommand_local = (
                    traincommand_local
                    + " "
                    + f"{baseName}_{str(trainingSetNum)}.{'tar' if not args.binary_training_optimization else 'bin'}"
                )

        trainFile.write(
            traincommand_local + "\n"
        )  # write the training command to the training command
        trainFile.write(
            "chmod -R 777 gradcam_plots saliency_maps *.log >> /dev/null 2>&1 \n"
        )  # change the permissions of the shell scripts to be executable.
        trainFile.write("echo end-is: `date` \n \n")  # add end timestamp
        training_batch_file.write(
            f"sbatch "
            f"-G {args.gpus if args.gpus > 0 else 1}"
            f" -c 8 "
            f" -n 4 "
            f" --mem=300G "
            f" --time={args.time_to_run_training} "
            f" -o {baseName}_trainlog_{str(dataset_num)}.log "
            f"{train_job_filename} "
            "\n"
        )  # add end timestamp to training file

    setNum = setNum + 1

training_batch_file.write(
    "echo end-is: `date` \n \n"
)  # add end timestamp to training file
training_batch_file.close()

logging.info("Done writing dataset and job files")
# change the permissions of the shell scripts to be executable.
os.system(f"chmod 777 {os.path.join(args.out_path, '*.log')} {os.path.join(args.out_path, '*.sh')}")
