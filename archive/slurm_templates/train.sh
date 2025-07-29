#!/usr/bin/bash -l

# This is meant to be used with slurm
# > sbatch -G 1 <this script>

# This file is a template with several variables that should be replaced.
# These template variables are:
# BIN_PATH The path to the VidActRecTrain.py script
# CHECKPOINT The path to use to save model checkpoints.
# BASE_DATA The path to the tar files that should be used for training and validation.
#           Data should be named BASE_DATA_1.tar, BASE_DATA_2.tar, etc
# MAX_FOLD The maximum fold to expect. BASE_DATA_1.tar through BASE_DATA_MAX_FOLD.tar should exist.
# LOG_FILE The path that should be used for the output logfile.
# WF_NAME The name of this workflow
#

#SBATCH --job-name=WF_NAME-training
#SBATCH --array=1-MAX_FOLD
#SBATCH --output=LOG_FILE.%A_%a

# This doesn't have webdataset installed
#export PATH=/koko/system/anaconda/envs/python39/bin:$PATH
export PATH=/koko/system/anaconda/envs/python38/bin:$PATH
echo start-is: `date`

cd BIN_PATH

# Create an eval string and a training string from the BASE_DATA path and the SLURM_ARRAY_TASK_ID
evaldata="BASE_DATA_${SLURM_ARRAY_TASK_ID}.tar"

# The training data is everything except for the
traindata=""
for I in `seq 1 $((${SLURM_ARRAY_TASK_ID}-1))`; do
    traindata+=" BASE_DATA_${I}.tar"
done

for I in `seq $((${SLURM_ARRAY_TASK_ID}+1)) MAX_FOLD`; do
    traindata+=" BASE_DATA_${I}.tar"
done

# Train
python3 VidActRecTrain.py --epochs 10 --template bees \
    --outname CHECKPOINT.${SLURM_ARRAY_TASK_ID} \
    --not_deterministic \
    --save_worst_n 100 \
    --evaluate $evaldata $traindata
succ=$?

echo end-is: `date`

# Success?
exit $succ

