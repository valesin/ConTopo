"""
Project-wide constants and configuration environment.
This file defines the default values for paths and other environment variables.
"""

import os

# Get the project root directory (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root directory for saving models and results
# Default: save/ResNet18/models
MODELS_ROOT = os.getenv("CONTOPO_MODELS_ROOT", os.path.join(_PROJECT_ROOT, "save/ResNet18/models"))

# Root directory for datasets
# Default: ./dataset
DATA_ROOT = os.getenv("CONTOPO_DATA_ROOT", os.path.join(_PROJECT_ROOT, "dataset"))

# Root directory for saving ensembles
# Default: save/ensembles
ENSEMBLES_ROOT = os.getenv("CONTOPO_ENSEMBLES_ROOT", os.path.join(_PROJECT_ROOT, "save/ensembles"))
