import os
import sys
import mlflow
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def setup_environment(
    config_name: str = "config", overrides: list[str] = None
) -> tuple[DictConfig, mlflow.entities.Experiment]:
    """
    Robust setup to use Hydra & MLflow configuration natively from interactive
    environments (Jupyter, Marimo) regardless of where the kernel launched.

    1. Finds the project root traversing upwards.
    2. Modifies `sys.path` and changes working directory to root safely.
    3. Handles Hydra's `HydraConfig` injection so resolvers like `${hydra:runtime.cwd}` work.
    4. Initializes Hydra configuration and returns configuration.

    Returns:
        tuple[DictConfig, Experiment]: The hydra configuration and the MLflow experiment.
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
    overrides = overrides or []
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
    from src.mlflow_utils import setup_mlflow

    setup_mlflow(cfg)

    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)

    return cfg, experiment
