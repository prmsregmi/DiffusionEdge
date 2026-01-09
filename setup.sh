#!/bin/bash
# DiffusionEdge Setup Script
# Downloads all pretrained models and sets up the environment

set -e

echo "=== DiffusionEdge Setup ==="

# Create checkpoints directory
mkdir -p checkpoints

# Download First Stage Model (required for all)
echo "Downloading First Stage Model..."
if [ ! -f checkpoints/first_stage_total_320.pt ]; then
    curl -L -o checkpoints/first_stage_total_320.pt https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/first_stage_total_320.pt
else
    echo "  First Stage Model already exists, skipping."
fi

# Download BSDS Model
echo "Downloading BSDS Model..."
if [ ! -f checkpoints/bsds.pt ]; then
    curl -L -o checkpoints/bsds.pt https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/bsds.pt
else
    echo "  BSDS Model already exists, skipping."
fi

# Download NYUD Model
echo "Downloading NYUD Model..."
if [ ! -f checkpoints/nyud.pt ]; then
    curl -L -o checkpoints/nyud.pt https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/nyud.pt
else
    echo "  NYUD Model already exists, skipping."
fi

# Download BIPED Model
echo "Downloading BIPED Model..."
if [ ! -f checkpoints/biped.pt ]; then
    curl -L -o checkpoints/biped.pt https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/biped.pt
else
    echo "  BIPED Model already exists, skipping."
fi

echo ""
echo "=== Setup Complete ==="
echo "Available models:"
ls -lh checkpoints/*.pt
echo ""
echo "Usage: uv run demo.py --model bsds --input_dir <input> --out_dir <output>"
