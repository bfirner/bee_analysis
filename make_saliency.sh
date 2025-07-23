#!/bin/bash

echo "Submitting Slurm job for saliency map generation..."
source ../../venv/bin/activate

# Submit the job with high resource allocation
sbatch \
  --job-name=saliency_generation \
  -n 3 \
  -c 5 \
  -G 2 \
  --mem=300G \
  --time=28800 \
  --output=saliency_job_%j.out \
  --error=saliency_job_%j.err \
  --o "dev/null"
  --wrap="
    echo 'Job started at: \$(date)'
    echo 'Working directory: \$(pwd)'
    echo 'Python path:'
    python -c 'import sys; print(\"\n\".join(sys.path))'
    
    # Activate virtual environment if needed
    source /research/projects/grail/dyd7/orig-files/venv/bin/activate
    
    # Change to the correct directory
    cd /research/projects/grail/dyd7/orig-files/Unified-bee-Runner/bee_analysis
    
    # Run the visualization generation with high resource settings
    python3 generate_visualizations.py \
        --checkpoint ../../model.checkpoint \
        --datasets ../../dataset_0.tar ../../dataset_1.tar ../../dataset_2.tar \
        --output_dir ./visualizations \
        --no_gradcam \
        --debug
        
    echo 'Job completed at: \$(date)'
  "

echo "Slurm submitted."