#!/bin/bash

# --- Configuration ---
PERSISTENT_DIR="./code"
CONTAINER_IMAGE="contopo.sif"
# Hardcoded output directory
OUTPUTS_DIR="/mnt/raid_storage/hasson/valerios/"

# 1. Ensure directories exist
echo "Ensuring persistent directories exist..."
mkdir -p "$PERSISTENT_DIR"
mkdir -p "$OUTPUTS_DIR"

# 2. Convert to absolute paths for Apptainer reliability
OUTPUTS_ABSOLUTE=$(realpath "$OUTPUTS_DIR")
PERSISTENT_ABSOLUTE=$(realpath "$PERSISTENT_DIR")

echo "------------------------------------------------------------"
echo "Container: $CONTAINER_IMAGE"
echo "Code Bind: $PERSISTENT_ABSOLUTE -> /persistent_repo/ConTopo/"
echo "Output Bind: $OUTPUTS_ABSOLUTE -> /persistent_repo/ConTopo/outputs"
echo "------------------------------------------------------------"

# 3. Launch the container
# Since there are no CLI arguments, we default to an interactive shell
echo "Launching interactive shell with GPU support..."

apptainer run --nv \
  --bind "$PERSISTENT_ABSOLUTE":/persistent_repo/ConTopo/ \
  --bind "$OUTPUTS_ABSOLUTE":/persistent_repo/ConTopo/outputs \
  "$CONTAINER_IMAGE" shell