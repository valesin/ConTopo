import os
from typing import Tuple

from configs import env


def generate_run_name(model_dir, trial):
    return f"{model_dir}___{trial}"


def get_trials(model_dir, trials):
    """Return list of trial names to use."""
    trial_path = os.path.join(env.MODELS_ROOT, model_dir)
    if not os.path.exists(trial_path):
         raise FileNotFoundError(f"Model directory not found: {trial_path}")
         
    all_trials = [d for d in os.listdir(trial_path) if os.path.isdir(os.path.join(trial_path, d)) and d.startswith("trial_")]
    if trials == "all":
        return sorted(all_trials)
    return [t for t in trials if t in all_trials]

def parse_run_name(run_name: str) -> Tuple[str, str]:
    """
    Parse the run name into model_dir and trial.
    Format is {model_dir}___{trial}.
    """
    if "___" not in run_name:
        raise ValueError(f"Invalid run_name format (missing separator '___'): {run_name}")
    
    # Split on the last occurrence of '___' as model_dir might contain it (though unlikely)
    parts = run_name.rsplit("___", 1)
    if len(parts) != 2:
        # Should be covered by the "___" check but for type safety
        raise ValueError(f"Could not parse run_name: {run_name}")
        
    model_dir, trial = parts
    return model_dir, trial