#!/usr/bin/env python3
"""
main.py — One-button orchestrator for the full ConTopo pipeline.

All scripts now use Hydra for config. Hydra overrides are passed through
as CLI arguments to child processes.

Runs all stages sequentially:
  0) Migrate existing legacy runs into MLflow
  1) Train models (via Hydra multirun)
  2) Cache inference artifacts
  3) Compute category similarity profiles
  3b) Compute per-model diagnostics (optional)
  4) Build ensembles
  4b) Compute ensemble diversity metrics (optional)
  4c) Compute ensemble RDM/RSA consistency (optional)
  5) Train adapters

Usage:
    python main.py                       # full pipeline
    python main.py --skip-migration      # skip step 0
    python main.py --skip-training       # skip steps 0+1
    python main.py --from-step 3         # start at step 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


SCRIPTS = [
    ("00_migrate_existing.py", "Migrating legacy runs to MLflow"),
    ("01_train_models.py",     "Training CE models"),
    ("02_cache_inference.py",  "Caching inference artifacts"),
    ("03_compute_profiles.py", "Computing category similarity profiles"),
    ("03b_compute_diagnostics.py", "Computing per-model diagnostics (optional)"),
    ("04_run_ensemble.py",     "Building ensembles"),
    ("04b_compute_diversity.py", "Computing ensemble diversity metrics (optional)"),
    ("04c_compute_consistency.py", "Computing ensemble RDM/RSA consistency (optional)"),
    ("05_train_adapters.py",   "Training adapters"),
]


def run_script(
    script_name: str,
    hydra_overrides: list[str] | None = None,
    cwd: str = ".",
) -> None:
    """Run a pipeline script with optional Hydra overrides."""
    script_path = os.path.join("scripts", script_name)
    cmd = [sys.executable, script_path] + (hydra_overrides or [])
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nERROR: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="ConTopo full-pipeline orchestrator.",
        epilog="All scripts use Hydra. Extra overrides are forwarded to each step.",
    )
    parser.add_argument("--from-step", type=int, default=0, help="Start from this step (0-5)")
    parser.add_argument("--skip-migration", action="store_true", help="Skip step 0")
    parser.add_argument("--skip-training", action="store_true", help="Skip steps 0 and 1")
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="Hydra overrides forwarded to every step (e.g. mlflow.tracking_uri=mlruns)",
    )
    args = parser.parse_args()

    start = args.from_step
    if args.skip_training:
        start = max(start, 2)
    elif args.skip_migration:
        start = max(start, 1)

    # Hydra overrides shared across all steps
    shared_overrides: list[str] = list(args.overrides)

    # Multirun sweep parameters for training
    training_multirun_overrides = [
        "--multirun",
        "loss.rho=0,0.008,0.04,0.2,1,5",
        "loss.topology=torus,grid",
        "trial=0,1,2,3,4",
    ]

    for idx, (script, desc) in enumerate(SCRIPTS):
        if idx < start:
            print(f"[SKIP] Step {idx}: {desc}")
            continue

        print(f"\n[STEP {idx}] {desc}")

        overrides: list[str] = list(shared_overrides)
        if idx == 1:
            # Training uses Hydra multirun
            overrides = training_multirun_overrides + overrides

        run_script(script, overrides)

    print(f"\n{'='*70}")
    print("  Pipeline complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
