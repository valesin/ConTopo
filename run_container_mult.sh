#!/bin/bash

# --- Configuration ---
PERSISTENT_DIR="./code"
# Hardcoded output directory
OUTPUTS_DIR="/mnt/raid_storage/hasson/valerios/"
CONTAINER_IMAGE="contopo.sif"

# 1. Ensure directories exist
mkdir -p "$PERSISTENT_DIR"
mkdir -p "$OUTPUTS_DIR"

# 2. Convert to absolute paths
OUTPUTS_ABSOLUTE=$(realpath "$OUTPUTS_DIR")
PERSISTENT_ABSOLUTE=$(realpath "$PERSISTENT_DIR")

echo "------------------------------------------------------------"
echo "Container: $CONTAINER_IMAGE"
echo "Code Bind: $PERSISTENT_ABSOLUTE -> /persistent_repo"
echo "Output Bind: $OUTPUTS_ABSOLUTE -> /persistent_repo/outputs"
echo "------------------------------------------------------------"

# 3. Launch the container (Simplified Binds)
echo "Launching interactive shell with GPU support..."

apptainer run --nv \
  --bind "$PERSISTENT_ABSOLUTE":/persistent_repo \
  --bind "$OUTPUTS_ABSOLUTE":/persistent_repo/outputs \
  "$CONTAINER_IMAGE" shell