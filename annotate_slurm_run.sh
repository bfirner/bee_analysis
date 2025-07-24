#!/bin/bash

echo "Submitting Slurm job for video annotation..."

# Submit the job with high resource allocation for video processing
sbatch \
  --job-name=video_annotation \
  -n 1 \
  -c 1 \
  -G 1 \
  --mem=200G \
  --time=28800 \
  -o "dev/null"\
  --wrap="#!/bin/bash
    export PATH=/usr/bin:/bin:/usr/local/bin:\$PATH
    
    echo 'Job started at: \$(date)'
    echo 'Running on node: \$(hostname)'
    echo 'Working directory: \$(pwd)'
    echo 'GPU info:'
    nvidia-smi || echo 'No GPU available'
    
    # Direct path to virtual environment python
    PYTHON_PATH='/research/projects/grail/dyd7/orig-files/venv/bin/python3'
    
    echo \"Python location: \$PYTHON_PATH\"
    \$PYTHON_PATH --version
    
    # Change to the correct directory
    cd /research/projects/grail/dyd7/orig-files/Unified-bee-Runner/bee_analysis || exit 1
    echo \"Changed to: \$(pwd)\"
    
    # Run the video annotation script
    \$PYTHON_PATH VidActRecAnnotate.py \
        --resume_from ../../model.checkpoint \
        --modeltype alexnet \
        --frames_per_sample 5 \
        --datalist ../../dataset.csv \
        --width 960 \
        --height 720
        
    echo 'Job completed at: \$(date)'
    echo 'Checking for output files:'
    ls -la annotated_*.mp4 || echo 'No annotated videos found'
    ls -la temp_frames/ || echo 'No temp frames directory found'
  "

echo "Slurm job submitted. Check status with: squeue -u \$USER"