"""
Run MLflow garbage collection to permanently delete runs in the deleted lifecycle stage.

Reads tracking URI from conf/mlflow/default.yaml and sets MLFLOW_TRACKING_URI
before invoking the mlflow gc command.

Usage:
    python scripts/mlflow_gc.py [--older-than 30d] [--dry-run]

Passes any extra arguments through to `mlflow gc`.
"""

import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def resolve_tracking_uri(project_root: Path) -> str:
    conf_dir = project_root / "conf" / "mlflow"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="default")

    # `cfg` is an OmegaConf object; access tracking_uri as a string
    raw_uri = str(cfg.get("tracking_uri"))
    # If Hydra runtime interpolation remains, substitute project root
    resolved = raw_uri.replace("${hydra:runtime.cwd}", str(project_root))
    return resolved


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = resolve_tracking_uri(project_root)

    print(f"Project root:  {project_root}")
    print(f"Tracking URI:  {tracking_uri}")
    print()

    # Set env var so mlflow gc can resolve artifact URIs
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri

    cmd = ["mlflow", "gc", "--backend-store-uri", tracking_uri, *sys.argv[1:]]
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
