#!/bin/bash

# Activate conda environment and run batch processing
source ~/miniconda3/bin/activate ram_dino_sam

echo "Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Run the batch processing script
python batch_process_videos.py

echo ""
echo "Batch processing completed!"
