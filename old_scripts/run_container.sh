#!/bin/bash

# --- Configuration ---
# Set ONE unified directory for your entire project (code + outputs).
# We are using your RAID storage since outputs are likely large.
PROJECT_DIR="/mnt/raid_storage/hasson/valerios/contopo_workspace"

CONTAINER_IMAGE="contopo.sif"

# 1. Ensure the directory exists
mkdir -p "$PROJECT_DIR"

# 2. Convert to absolute path
PROJECT_ABSOLUTE=$(realpath "$PROJECT_DIR")

echo "------------------------------------------------------------"
echo "Container: $CONTAINER_IMAGE"
echo "Project Bind: $PROJECT_ABSOLUTE -> /persistent_repo"
echo "------------------------------------------------------------"

# 3. Launch the container (Single Bind)
echo "Launching interactive shell with GPU support..."

apptainer run --nv \
  --bind "$PROJECT_ABSOLUTE":/persistent_repo \
  "$CONTAINER_IMAGE" shell