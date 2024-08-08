#!/bin/bash

# TODO: ADD some dynamic stuff here, like the path to the python script

# QUICKSTART OPTIONS, or some stuff that could be interesting to test out
# --width: width of the video
# --height: height of the video
# --number-of-samples: number of samples to take by the video sampler max for each line in the separated dataset
# --frames-per-sample: number of frames to take for each sample
# --start: the step with which to start
# --end: the step with which to end

# Additionally, you can run the following command to get the help menu
# python Unified-bee-Runner/master_run.py --help

# restrict the number of subthreads because all our commands use multiprocessing
# this is to prevent the system from running out of memory
# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python Unified-bee-Runner/master_run.py --height 720 --width 960 --number-of-samples 40000 --frames-per-sample 5 >>BEERUN.log 2>&1
