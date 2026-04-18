"""
Run with:  uv run python notebooks/debug_setup_timing.py
"""

import os
import sys
import time


def ts(label, t0):
    print(f"  [{time.time() - t0:.3f}s] {label}")


t0 = time.time()

# ── 1. Root discovery ──────────────────────────────────────────────────────────
root_dir = os.getcwd()
while root_dir != "/" and not os.path.exists(
    os.path.join(root_dir, "conf", "config.yaml")
):
    root_dir = os.path.dirname(root_dir)
ts("root discovery", t0)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.chdir(root_dir)
ts("sys.path + chdir", t0)

# ── 2. Hydra compose ──────────────────────────────────────────────────────────
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig

ts("hydra imports", t0)

config_dir = os.path.join(root_dir, "conf")
overrides = ["groups=samples9"]

with initialize_config_dir(version_base=None, config_dir=config_dir):
    ts("initialize_config_dir", t0)
    cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    ts("compose", t0)
    HydraConfig.instance().set_config(cfg)
    ts("HydraConfig.set_config", t0)

# ── 3. MLflow setup ───────────────────────────────────────────────────────────
import mlflow

ts("mlflow import", t0)

from src.mlflow_utils import apply_mlflow_env_overrides, setup_mlflow

ts("src imports", t0)

apply_mlflow_env_overrides(cfg)
ts("apply_mlflow_env_overrides", t0)

mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
ts("mlflow.set_tracking_uri", t0)

exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
ts("mlflow.get_experiment_by_name", t0)

# ── 4. Repository + experiment lookup ─────────────────────────────────────────
from src.repositories.functional_run_repository import configure_run_repository

configure_run_repository(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)
ts("configure_run_repository", t0)

# ── 5. Signature (first call, imports cold) ───────────────────────────────────
from src.ensemble.selector import encode_groups_signature

sig = encode_groups_signature(cfg.groups)
ts(f"encode_groups_signature → {sig}", t0)

# ── 6. Simulate dropdown re-run cost (compose_groups only) ────────────────────
t1 = time.time()
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg2 = compose(
        config_name="config", overrides=["groups=default"], return_hydra_config=True
    )
    HydraConfig.instance().set_config(cfg2)
sig2 = encode_groups_signature(cfg2.groups)
print(
    f"\n  [{time.time() - t1:.3f}s] compose_groups re-run cost (dropdown change simulation)"
)
print(f"  signature → {sig2}")

print(f"\nTotal (cold): {time.time() - t0:.3f}s")
