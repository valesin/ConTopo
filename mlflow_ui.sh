#!/usr/bin/env bash
set -euo pipefail

BACKEND_URI="${MLFLOW_TRACKING_URI:-sqlite:///outputs/mlflow.db}"
PORT="${MLFLOW_UI_PORT:-5000}"

echo "MLflow UI → backend: $BACKEND_URI  port: $PORT"
uv run mlflow ui --backend-store-uri "$BACKEND_URI" --port "$PORT"
