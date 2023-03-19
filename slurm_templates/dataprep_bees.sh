#!/usr/bin/bash -l

# This is meant to be used with slurm
# > sbatch -G 1 <this script>

# This file is a template with several variables that should be replaced.
# These template variables are:
# BIN_PATH The path to the VidActRecDataprep.py script
# CSV_BASE The path to the csv file(s) that should be used as input, less the _FOLD.csv suffix.
# TAR_BASE The path to the tar file that should be used as output, less the _FOLD.tar suffix.
# MAX_FOLD The maximum fold to expect. CSV_BASE_1.csv through CSV_BASE_MAX_FOLD.csv should exist.
# LOG_FILE The path that should be used for the output logfile.
# OUT_PATH The path for the output tar files.
# WF_NAME The name of this workflow

#SBATCH --job-name=WF_NAME-dataprep
#SBATCH --array=1-MAX_FOLD
#SBATCH --output=LOG_FILE.%A_%a

# This installation has ffmpeg-python installed
export PATH=/koko/system/anaconda/envs/python38/bin:$PATH
echo start-is: `date`

cd OUT_PATH

in_csv="CSV_BASE_${SLURM_ARRAY_TASK_ID}.csv"
out_tar="TAR_BASE_${SLURM_ARRAY_TASK_ID}.tar"

# Crop 400x400 patches with 20 pixels of crop noise to resist "tells" from
# motion caused by changes in object positions.
python3 BIN_PATH/VidActRecDataprep.py --width 400 --height 400 --samples 500 --crop_noise 20 --out_channels 1 --frames_per_sample 1 "${in_csv}" "${out_tar}"
succ=$?

echo end-is: `date`

# Success?
exit $succ

