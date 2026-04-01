#!/bin/bash
# launch_sweep.sh — Submits one SkyPilot job per (rho, trial) combination.
# Each job runs on a single Spot GPU and trains one model configuration.
#
# Usage:
#   source .env.secrets && ./launch_sweep.sh
#
# To submit a subset, edit RHOS or TRIALS below before running.
#
# Prerequisites:
#   - sky is installed and configured
#   - .env.secrets is sourced in the current shell (secrets are read from env vars)
#   - Docker image is built and accessible (update sky_task.yaml with image_id)

set -e

RHOS=(0.0 0.008 0.04 0.2 1.0 5.0)
TRIALS=(0 1 2 3 4 5 6 7 8 9)

SECRETS=(
  --secret AWS_ACCESS_KEY_ID
  --secret AWS_SECRET_ACCESS_KEY
  --secret MLFLOW_S3_ENDPOINT_URL
  --secret MLFLOW_TRACKING_USERNAME
  --secret MLFLOW_TRACKING_PASSWORD
)

SUBMITTED=0

for RHO in "${RHOS[@]}"; do
  for TRIAL in "${TRIALS[@]}"; do
    JOB_NAME="contopo-rho${RHO}-trial${TRIAL}"
    echo "Submitting $JOB_NAME..."
    sky jobs launch sky_task.yaml \
      --name "$JOB_NAME" \
      --env LOSS_RHO="$RHO" \
      --env TRIAL="$TRIAL" \
      "${SECRETS[@]}" \
      -y
    SUBMITTED=$((SUBMITTED + 1))
  done
done

echo ""
echo "Done. Submitted ${SUBMITTED} jobs."
echo "Monitor with: sky jobs queue"
