#!/bin/bash
# launch_sweep.sh — Submits one SkyPilot job per (rho, trial) combination.
# Each job runs on a single Spot GPU and trains one model configuration.
#
# Usage:
#   ./launch_sweep.sh                  # submit all 60 jobs
#   RHOS=(0.0 0.008) ./launch_sweep.sh # submit a subset of rho values
#   TRIALS=(0 1 2) ./launch_sweep.sh   # submit a subset of trials
#
# Prerequisites:
#   - sky is installed and configured
#   - ~/.env.secrets.contopo exists with MLflow + S3 credentials
#   - Docker image is built and pushed to a registry accessible by SkyPilot
#     (update sky_task.yaml with the image name/tag before running)

set -e

RHOS=(0.0 0.008 0.04 0.2 1.0 5.0)
TRIALS=(0 1 2 3 4 5 6 7 8 9)

SUBMITTED=0
SKIPPED=0

for RHO in "${RHOS[@]}"; do
    for TRIAL in "${TRIALS[@]}"; do
        JOB_NAME="contopo-rho${RHO}-trial${TRIAL}"
        echo "Submitting $JOB_NAME..."
        sky jobs launch sky_task.yaml \
            --name "$JOB_NAME" \
            --env LOSS_RHO="$RHO" \
            --env TRIAL="$TRIAL" \
            -y
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "Done. Submitted ${SUBMITTED} jobs."
echo "Monitor with: sky jobs queue"
