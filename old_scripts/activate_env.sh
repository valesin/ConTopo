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
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SECRETS_FILE="$SCRIPT_DIR/.env.secrets"

    if [ -f "$SECRETS_FILE" ]; then
        set -a
        source "$SECRETS_FILE"
        set +a
        echo "✓ Loaded environment secrets from .env.secrets"
    else
        echo "⚠ No .env.secrets found (MLflow/S3 auth env vars not loaded)"
    fi

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
