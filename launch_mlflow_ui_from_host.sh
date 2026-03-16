#!/bin/bash

# Launch MLflow UI inside the container from the host (non-interactive)
# This script runs launch_mlflow_ui_container.sh inside the container without entering the shell

CONTAINER_IMAGE="contopo.sif"
PROJECT_DIR="/mnt/raid_storage/hasson/valerios/contopo_workspace"
SCRIPT_PATH="/persistent_repo/ConTopo/launch_mlflow_ui_container.sh"

apptainer exec --nv \
  --bind "$PROJECT_DIR":/persistent_repo \
  "$CONTAINER_IMAGE" \
  "$SCRIPT_PATH"
