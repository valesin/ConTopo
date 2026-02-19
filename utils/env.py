"""
Utility to access project configuration and environment variables.

This module provides a centralized way to access paths and constants defined in
`configs/env.py`. It also serves as a guide for expanding the configuration
and substituting hardcoded values in the codebase.

Usage:
    from utils import env
    
    model_path = os.path.join(env.MODELS_ROOT, "my_model")

--------------------------------------------------------------------------------
INSTRUCTIONS FOR EXPANDING CONFIGURATION
--------------------------------------------------------------------------------

To add a new configuration variable:

1.  Open `configs/env.py`.
2.  Add a new constant with a descriptive name (upper case).
3.  Use `os.getenv("CONTOPO_YOUR_VAR_NAME", default_value)` to allow
    overriding via environment variables.
    
    Example:
    # configs/env.py
    MY_NEW_PATH = os.getenv("CONTOPO_MY_NEW_PATH", "./default/path")

4.  The new variable will be automatically available in `utils.env` via
    the import in this file.

--------------------------------------------------------------------------------
INSTRUCTIONS FOR SUBSTITUTING HARDCODED VALUES
--------------------------------------------------------------------------------

If you find a hardcoded path or value in the codebase that should be configurable:

1.  Identify the value (e.g., "save/ResNet18/models").
2.  Check if a corresponding constant already exists in `configs/env.py`.
    - If yes, proceed to step 3.
    - If no, follow the "INSTRUCTIONS FOR EXPANDING CONFIGURATION" above to add it.
3.  Import `env` in the target file:
    
    from utils import env

4.  Replace the hardcoded value with `env.CONSTANT_NAME`.

    Example (in utils/ensemble_utils.py):
    
    # BEFORE:
    MODELS_ROOT = "save/ResNet18/models"
    
    # AFTER:
    from utils import env
    MODELS_ROOT = env.MODELS_ROOT

"""

# Re-export everything from configs.env
from configs.env import *
