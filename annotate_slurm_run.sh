#!/bin/bash

echo "Submitting Slurm job for video annotation..."

# Submit the job with high resource allocation for video processing
sbatch \
  --job-name=video_annotation \
  -n 3 \
  -c 8 \
  -G 3 \
  --mem=200G \
  --time=28800 \
  -o "annotation_job.out" \
  -e "annotation_job.err" \
  --wrap="#!/bin/bash
    export PATH=/usr/bin:/bin:/usr/local/bin:\$PATH
    
    echo 'Job started at:' \$(date)
    echo 'Running on node:' \$(hostname)
    echo 'Working directory:' \$(pwd)
    echo 'GPU info:'
    nvidia-smi || echo 'No GPU available'
    
    # Change to the correct directory first
    cd /research/projects/grail/ex28/refined_real_ant_crop/Unified-bee-Runner/bee_analysis || exit 1
    echo 'Changed to:' \$(pwd)
    
    # Activate virtual environment using . instead of source for sh compatibility
    . ../../venv/bin/activate || exit 1
    echo 'Virtual environment activated'
    echo 'Python location:' \$(which python3)
    echo 'Python version:' \$(python3 --version)
    
    # Install missing package if needed
    pip install ffmpeg-python || echo 'ffmpeg-python already installed'
    
    # List files to verify they exist
    echo 'Checking for required files:'
    ls -la ../../model.checkpoint || echo 'model.checkpoint not found'
    ls -la ../../datasetcopy.csv || echo 'datasetcopy.csv not found'
    ls -la VidActRecAnnotate.py || echo 'VidActRecAnnotate.py not found'
    
    # Check video files exist
    echo 'Checking video files from CSV:'
    python3 -c \"
import csv
with open('../../datasetcopy.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        video_path = row[0]
        import os
        if os.path.exists(video_path):
            print(f'✓ {video_path} exists')
        else:
            print(f'✗ {video_path} NOT FOUND')
\"
    
    # Run the video annotation script - all logging goes to annotations.log via the script
    echo 'Starting video annotation...'
    python3 VidActRecAnnotate.py \
        --resume_from ../../model.checkpoint \
        --modeltype alexnet \
        --frames_per_sample 5 \
        --datalist ../../datasetcopy.csv \
        --width 740 \
        --height 400 \
        --crop_x_offset -20 \
        --crop_y_offset 20 \
        --debug \
        --flip 1
    
    echo 'Job completed at:' \$(date)
    echo 'Checking for output files:'
    ls -la annotated_*.mp4 || echo 'No annotated videos found'
    ls -la temp_frames/ || echo 'No temp frames directory found'
    ls -la ../../annotations.log || echo 'No log file found'
  "

echo "Slurm job submitted. Check status with: squeue -u \$USER"
echo "Output will be in: annotation_job.out and annotation_job.err"
echo "Log file will be at: ../../annotations.log"