# bee_analysis

Analyze some bee videos. Could also work on other data too.

## Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt).

## Uses for Important files

- [`make_validation_training.py`](make_validation_training.py): creates `train_*.sh` files and `training_run.sh` files that are used for training the dataset.
- [`VidActRecAnnotate.py`](VidActRecAnnotate.py): annotate a video with what the model is thinking
- [`VidActRecLabel.py`](VidActRecLabel.py): easily create a bounding box for generating crop offsets

## Training a model

Train a model with the VidActRecTrain.py script. For example:

> python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar train.tar

If you are doing cross validation, you can specify multiple tar files for training as the last
arguments:

> python3 VidActRecTrain.py --epochs 10 --modeltype alexnet --evaluate eval.tar a.tar b.tar c.tar

By default the model is saved to `model.checkpoint`, but that can be changed with the `--outname`
option.

Run the script with `--help` for more details.

### N-Fold Cross Validation

We would like to believe that all of our data is homegenous, but reality is often different. It is
possible that when you split your data into a training set and a validation set you will put all of
the hard data into the training side and leave the validation set too easy. The opposite and many
more permutations are also possible.

To prevent being lead astray by troublesome data many researchers use _n-fold cross validation_.
This means that you split your data into _n_ separate chunks. Then you run _n_ experiments, with a
different chunk being used as the validation set during each experiment. All of the chunks not being
used for validation are used for training.

If your results are consistent across all of the experiments then your data is consistent. If your
results are inconsistent this doesn't mean that your approach or data are bad, but it does mean that
you have something more complicated happening and you will have to look into your data to understand
what is happening.

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

```
python3 VidActRecAnnotate.py --datalist <csv used in dataprep> \
                             --resume_from <training checkpoint> \
                             --modeltype <model type of checkpoint>
```

You can also specify the `--class_names` option to set strings for the class names.

Use the `--label_classes` option to set the number of classes predicted by the model.

As is usual, you can run the script with the `--help` option for more details.
