#!/bin/bash

# Activate the ConTopo conda environment
# Usage: source activate_env.sh

# Initialize conda in bash if not already initialized
if [ -z "$CONDA_SHLVL" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Activate the environment
conda activate contopo

if [ $? -eq 0 ]; then
    echo "✓ ConTopo conda environment activated"
    echo "Environment: $(basename $CONDA_PREFIX)"
    echo "Python: $(python --version)"
    echo ""
    echo "Available tools:"
    echo "  - PyTorch (GPU ready with CUDA 12.1)"
    echo "  - MLflow (experiment tracking)"
    echo "  - Jupyter/IPython (interactive notebooks)"
    echo "  - Polars (fast data processing)"
    echo "  - pytest (testing)"
else
    echo "✗ Failed to activate contopo environment"
    echo "Make sure you've created the environment with: conda env create -f environment.yml"
    return 1
fi
