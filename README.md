# bee_analysis
Analyze some bee videos. Could also work on other data too.

## Dependencies

Torch, torchvision, ffmpeg, and webdataset.

In general, install through pip3 or conda:

> pip3 install torch torchvission webdataset ffmpeg

See https://pytorch.org/get-started/locally/ for additional installation instructions

## Creating a dataset

First run `make_train.sh` to create a csv file with labels for each video.
> bash make_train.sh *path/to/videos* > dataset.csv

The file paths created by `make_train_csv.sh` are relative so it should be run from the same
directory as the dataprep step will be run. 

Next process that csv file with VidActRecDataprep.py. For example:
> python3 python3 VidActRecDataprep.py --width 400 --height 400 --resizeestrategy crop --samples 500 --out_channels 1 --frames_per_sample 1 dataset.csv dataset.tar

The script can also be run with `--help` for more details.

## Training a model

Train a model with the VidActRecTrain.py script. For example:
> python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar train.tar

By default the model is saved to `model.checkpoint`, but that can be changed with the `--outname`
option. Run the script with `--help` for more details.
