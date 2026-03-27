#!/bin/bash

# ConTopo conda environment setup script
# Creates the conda environment from environment.yml
# Usage: ./setup_env.sh [--activate]

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.yml"
ACTIVATE_AFTER=false

# Parse arguments
if [ "$1" == "--activate" ]; then
    ACTIVATE_AFTER=true
fi

echo "================================================"
echo "ConTopo Conda Environment Setup"
echo "================================================"
echo ""

# Initialize conda in bash if not already initialized
if [ -z "$CONDA_SHLVL" ]; then
    echo "Initializing conda for bash..."
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "✗ Error: conda not found"
    echo ""
    echo "Please install conda first:"
    echo "  - Miniconda: https://docs.conda.io/projects/miniconda/en/latest/"
    echo "  - Anaconda: https://www.anaconda.com/download"
    exit 1
fi

echo "✓ conda found: $(conda --version)"
echo ""

# Check if environment.yml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "✗ Error: environment.yml not found at $ENV_FILE"
    exit 1
fi

echo "✓ environment.yml found"
echo ""

# Check if environment already exists
if conda env list | grep -q "^contopo "; then
    echo "⚠ Environment 'contopo' already exists"
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment..."
        conda env update -f "$ENV_FILE" --prune
        echo "✓ Environment updated"
    else
        echo "Skipping update"
        ACTIVATE_AFTER=true  # Still offer to activate
    fi
else
    echo "Creating conda environment from $ENV_FILE..."
    conda env create -f "$ENV_FILE"
    echo ""
    echo "✓ Environment 'contopo' created successfully"
fi

# Install the project itself as editable (required for 'src' imports)
echo "Installing ConTopo package (editable)..."
conda run -n contopo pip install -e "$SCRIPT_DIR" --no-deps --quiet
echo "✓ ConTopo package installed"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  source activate_env.sh"
echo ""
echo "Or add this alias to ~/.bashrc or ~/.zshrc:"
echo "  alias contopo='source $SCRIPT_DIR/activate_env.sh'"
echo ""

# Activate if requested
if [ "$ACTIVATE_AFTER" = true ]; then
    echo "Activating environment..."
    conda activate contopo
    echo ""
    echo "✓ Environment activated!"
    echo "Environment: $(basename $CONDA_PREFIX)"
    echo "Python: $(python --version)"
fi
