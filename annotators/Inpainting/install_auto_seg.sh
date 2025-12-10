#!/bin/bash

# Automatic Installation Script for RAM + Grounding DINO + SAM2 Pipeline
# This script will download all necessary models and install dependencies

set -e  # Exit on error

echo "================================================"
echo "Installing Automatic Segmentation Pipeline"
echo "================================================"

# Create directories
mkdir -p models
cd models

# Install Python dependencies
echo ""
echo "[1/5] Installing base dependencies..."
pip install torch torchvision opencv-python pillow numpy tqdm supervision matplotlib scipy timm transformers

# Install SAM2
echo ""
echo "[2/5] Installing SAM2..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download SAM2 checkpoints
echo ""
echo "[3/5] Downloading SAM2 checkpoints..."
if [ ! -f "sam2_hiera_large.pt" ]; then
    echo "Downloading SAM2 Large..."
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
fi

if [ ! -f "sam2_hiera_base_plus.pt" ]; then
    echo "Downloading SAM2 Base+..."
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
fi

# Install and setup Grounding DINO
echo ""
echo "[4/5] Installing Grounding DINO..."
pip install groundingdino-py

# Download Grounding DINO checkpoint
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    echo "Downloading Grounding DINO checkpoint..."
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

# Download Grounding DINO config
if [ ! -f "GroundingDINO_SwinT_OGC.py" ]; then
    echo "Downloading Grounding DINO config..."
    wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
fi

# Install RAM (optional)
echo ""
echo "[5/5] Installing RAM (optional - for automatic tag generation)..."
read -p "Do you want to install RAM for automatic tag generation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Clone and install RAM
    if [ ! -d "recognize-anything" ]; then
        git clone https://github.com/xinyu1205/recognize-anything.git
        cd recognize-anything
        pip install -e .
        cd ..
    fi
    
    # Download RAM checkpoint
    if [ ! -f "ram_plus_swin_large_14m.pth" ]; then
        echo "Downloading RAM checkpoint..."
        wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
    fi
    echo "RAM installed successfully!"
else
    echo "Skipping RAM installation. You can use manual text prompts instead."
fi

cd ..

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Model checkpoints are in: ./models/"
echo ""
echo "Quick Start Examples:"
echo ""
echo "1. With RAM (fully automatic):"
echo "   python automatic_segmentation.py \\"
echo "       --input image.jpg \\"
echo "       --mode image \\"
echo "       --sam2-checkpoint models/sam2_hiera_large.pt \\"
echo "       --grounding-dino-config models/GroundingDINO_SwinT_OGC.py \\"
echo "       --grounding-dino-checkpoint models/groundingdino_swint_ogc.pth \\"
echo "       --ram-checkpoint models/ram_plus_swin_large_14m.pth"
echo ""
echo "2. Without RAM (manual prompts):"
echo "   python automatic_segmentation.py \\"
echo "       --input image.jpg \\"
echo "       --mode image \\"
echo "       --sam2-checkpoint models/sam2_hiera_large.pt \\"
echo "       --grounding-dino-config models/GroundingDINO_SwinT_OGC.py \\"
echo "       --grounding-dino-checkpoint models/groundingdino_swint_ogc.pth \\"
echo "       --text-prompt 'person . car . dog' \\"
echo "       --no-ram"
echo ""
echo "See README_AUTO_SEGMENTATION.md for more examples!"
