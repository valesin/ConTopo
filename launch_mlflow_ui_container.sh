#!/bin/bash

# Launch MLflow UI inside the container
# This script ensures MLflow uses the correct absolute path for the database file
# The database will be located at /persistent_repo/ConTopo/outputs/mlflow.db

mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:////persistent_repo/ConTopo/outputs/mlflow.db
