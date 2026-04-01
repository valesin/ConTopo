#!/bin/bash

# Launch MLflow UI inside the container from the host (non-interactive)
# This script runs launch_mlflow_ui_container.sh inside the container without entering the shell

# Use absolute path for container image
CONTAINER_IMAGE="$(realpath ./contopo.sif)"
PROJECT_DIR="/mnt/raid_storage/hasson/valerios/contopo_workspace"

# Convert to absolute path for bind mount
PROJECT_ABSOLUTE=$(realpath "$PROJECT_DIR")

echo "------------------------------------------------------------"
echo "Container: $CONTAINER_IMAGE"
echo "Project Bind: $PROJECT_ABSOLUTE -> /persistent_repo"
echo "------------------------------------------------------------"

SCRIPT_PATH="/persistent_repo/ConTopo/launch_mlflow_ui_container.sh"

apptainer exec --nv \
  --bind "$PROJECT_ABSOLUTE":/persistent_repo \
  "$CONTAINER_IMAGE" \
  "$SCRIPT_PATH"
