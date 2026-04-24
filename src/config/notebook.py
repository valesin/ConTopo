import os
import sys
import mlflow
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import Experiment
from omegaconf import DictConfig


def setup_environment(
    config_name: str = "config",
    overrides: list[str] | None = None,
    mlflow_config: str = "default",
) -> tuple[DictConfig, Experiment]:
    """
    Robust setup for interactive environments (Jupyter, Marimo).

    1. Finds the project root traversing upwards.
    2. Modifies sys.path and chdirs to root so imports and artifact paths resolve correctly.
    3. Injects HydraConfig so ${hydra:runtime.cwd} resolvers work.
    4. Composes Hydra config and initialises MLflow.

    mlflow_config selects conf/mlflow/<name>.yaml:
      "default"   → sqlite:///outputs/mlflow.db  (local DB used by pipeline scripts)
      "notebook"  → http://localhost:5000  (remote server via SSH tunnel)
    To use a remote tracking server without changing this value, set MLFLOW_TRACKING_URI
    (and optionally MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD) in the environment.
    Additional Hydra overrides can be passed via overrides list.

    Returns:
        tuple[DictConfig, Experiment]: resolved config and the MLflow experiment object.
    """
    # 1. Start from current execution directory and walk up to find project root
    root_dir = os.getcwd()
    while root_dir != "/" and not os.path.exists(
        os.path.join(root_dir, "conf", "config.yaml")
    ):
        root_dir = os.path.dirname(root_dir)

    if root_dir == "/":
        raise FileNotFoundError(
            "Could not find the ConTopo project root (missing conf/config.yaml)."
        )

    # 2. Add to Python Path and explicitly switch there for smooth imports/artifact behaviour
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    os.chdir(root_dir)

    # 3. Compose Configuration
    overrides = list(overrides or [])
    if not any(o.startswith("mlflow=") for o in overrides):
        overrides = [f"mlflow={mlflow_config}"] + overrides
    config_dir = os.path.join(root_dir, "conf")

    # We use initialize_config_dir (which is part of the valid Compose API initialization methods)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # We must pass return_hydra_config=True to fully hydrate the configuration
        cfg = compose(
            config_name=config_name, overrides=overrides, return_hydra_config=True
        )

        # VERY IMPORTANT: When using the Compose API, Hydra does not automatically set the
        # global HydraConfig instance. Any interpolation heavily relying on `${hydra:...}`
        # resolvers will crash with "HydraConfig was not set". We must inject it manually.
        HydraConfig.instance().set_config(cfg)

    # 4. Bring up MLflow using existing utility
    # (requires absolute import after sys.path is updated)
    from src.mlflow_utils import apply_mlflow_env_overrides, setup_mlflow
    from src.repositories.functional_run_repository import (
        configure_artifact_cache_dir,
        configure_run_repository,
    )

    apply_mlflow_env_overrides(cfg)
    setup_mlflow(cfg)
    configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)
    configure_artifact_cache_dir(cfg.mlflow.artifact_cache_dir)

    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{cfg.mlflow.experiment_name}' not found")

    return cfg, experiment


def compose_groups(groups_name: str) -> DictConfig:
    """
    Lightweight Hydra compose returning just cfg.groups for a named groups config.

    Intended for reactive notebook cells that only need to recompute a groups
    signature when a dropdown changes — without re-running MLflow setup.
    Must be called after setup_environment() (relies on cwd and sys.path already set).
    """
    root_dir = os.getcwd()
    config_dir = os.path.join(root_dir, "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[f"groups={groups_name}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
    return cfg.groups
