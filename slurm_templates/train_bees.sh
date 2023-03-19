#!/usr/bin/bash -l

# This is meant to be used with slurm
# > sbatch -G 1 <this script>

# This file is a template with several variables that should be replaced.
# These template variables are:
# BIN_PATH The path to the VidActRecDataprep.py script
# TRAIN_PATH The path to the tar file(s) that should be used for training input.
# EVAL_PATH The path to the tar file that should be used for validation.
# LOG_FILE The path that should be used for the output logfile.
# CHECKPOINT The path to use to save model checkpoints.
#

#SBATCH --output=LOG_FILE

# This doesn't have webdataset installed
#export PATH=/koko/system/anaconda/envs/python39/bin:$PATH
export PATH=/koko/system/anaconda/envs/python38/bin:$PATH
echo start-is: `date`

cd BIN_PATH

# Train
python3 VidActRecTrain.py --epochs 10 --template bees \
    --outname CHECKPOINT \
    --not_deterministic \
    --save_worst_n 100 \
    --evaluate EVAL_PATH TRAIN_PATH
succ=$?

echo end-is: `date`

# Success?
exit $succ

