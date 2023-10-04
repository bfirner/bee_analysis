# bee_analysis
Analyze some bee videos. Could also work on other data too.

## Dependencies

Torch, torchvision, ffmpeg (or possibly ffmpeg-python), and webdataset.

In general, install through pip3 or conda:

> pip3 install torch torchvision webdataset ffmpeg

See https://pytorch.org/get-started/locally/ for additional installation instructions

## Creating a dataset

### Create a dataset csv

First run `make_train_csv.sh` to create a csv file with labels for each video.
> bash make_train_csv.sh *path/to/videos* > dataset.csv

The `make_train_csv.sh` shell script expects to find three files in the directory named:
* logNeg.txt
* logNo.txt
* logPos.txt

Each of those files should have a single column of text that specifies the beginning timestamps of
events. In initial experiements those events were negative polarity, no magnet, and positive. The
same file could be used for any type of data though, the training code is agnostic to the input
classes. The date format in the files should be: `YYYYMMDD_HHMMSS`

The file paths created by `make_train_csv.sh` are relative so it should be run from the same
directory as the dataprep step will be run.

### Make scripts to scale data preparation, training and evaluation on servers that run slurm 

Slurm is a system for managing batch jobs and GPU allocation. 

After to you make the dataset.csv as above, run the 'make_validation_training.py' script with the dataset.csv as input. This will
create smaller dataset tar files and scripts to do K-cross fold validation using sbatch and srun.

#### Synthetic Roach Data

`roach_csv.py` takes the place of `make_train_csv.sh` for the synthetic roach data.

### Process the csv into training and/or testing data.

Next process that csv file with VidActRecDataprep.py. For example:
> python3 VidActRecDataprep.py --width 400 --height 400 --samples 500 --crop_noise 20 --out_channels 1 --frames_per_sample 1 dataset.csv dataset.tar

The video can also be scaled with the `--scale` option:
> python3 VidActRecDataprep.py --width 200 --height 200 --scale 0.5 --samples 500 --crop_noise 20 --out_channels 1 --frames_per_sample 1 dataset.csv dataset.tar

Cropping is done after scaling, so a width and height of 200 after scaling by 0.5 will cover the
same pixels as a crop of 400 by 400 without any scaling. To shift the location of the training crop,
use the `--crop_x_offset` and `--crop_y_offset` command line arguments.


The `samples` value should be as large as is reasonable. If your data has little movement from one
frame to the next then you should sample sparsely to avoid having redundant frames. However, if your
video is very dynamic there is nothing wrong with sampling all of the frames. If you specify more
frames than are in a video then VidActRecDataprep.py will simply sample all available frames.

The `crop_nose` option adds some randomness to the cropping location, which is important to prevent
overfitting when there is some obvious visual tell in the data (for example if the camera is slowly
changing position over time).

The script can also be run with `--help` for more details.

#### N-Fold Cross Validation

We would like to believe that all of our data is homegenous, but reality is often different. It is
possible that when you split your data into a training set and a validation set you will put all of
the hard data into the training side and leave the validation set too easy. The opposite and many
more permutations are also possible.

To prevent being lead astray by troublesome data many researchers use *n-fold cross validation*.
This means that you split your data into *n* separate chunks. Then you run *n* experiments, with a
different chunk being used as the validation set during each experiment. All of the chunks not being
used for validation are used for training.

If your results are consistent across all of the experiments then your data is consistent. If your
results are inconsistent this doesn't mean that your approach or data are bad, but it does mean that
you have something more complicated happening and you will have to look into your data to understand
what is happening.

If you are going to use n-fold cross validation then be sure to break your dataset into *n* chunks
at this point, which means running `VidActRecDataprep.py` *n* times.

## Training a model

Train a model with the VidActRecTrain.py script. For example:
> python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar train.tar

If you are doing cross validation, you can specify multiple tar files for training as the last
arguments:
> python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar a.tar b.tar c.tar

By default the model is saved to `model.checkpoint`, but that can be changed with the `--outname`
option.

Run the script with `--help` for more details.

### Training with Arbitrary Targets and Inputs

Once a webdataset is built, each file in the dataset can be used as an input or an output. To use
one of the files as an input, use the `--label` option:
> --labels <input name 1> <input name 2> ...

If the inputs should be converted to one-hot vectors then set `--convert_idx_to_classes 1`

Currently mixing one-hot with other inputs is not supported, either all labels will be treated a
one-hot or all labels will be taken as-is.

Inputs, meaning vector inputs or single inputs, can be specified with the `--vector_input` argument.
This takes a list of names and works in the same way as the `--labels` option.

### Training with Regression Targets

To train with regression targets, first select a regression loss function, such as mean square
error:
> --loss_fun MSELoss

Make sure to set `--convert_idx_to_classes 0`

Regression training is sensitive to differences in magnitude of the training targets. For example,
if one regression target has values an order of magnitude, then its loss will also be an order of
magnitude larger which could suppress learning of the smaller target. To normalize this, use the
`--normalize_outputs` option.


## Annotating a Video

Once a model has been trained you may want to see the visualization of feature intensity along with
the class predictions per frame.

Annotation is done with the `VidActRecAnnotate.py` script.

~~~~
python3 VidActRecAnnotate.py --datalist <csv used in dataprep> \
                             --resume_from <training checkpoint> \
                             --modeltype <model type of checkpoint>
~~~~

You can also specify the `--class_names` option to set strings for the class names.

Use the `--label_classes` option to set the number of classes predicted by the model.

As is usual, you can run the script with the `--help` option for more details.  
