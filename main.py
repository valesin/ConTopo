#!/usr/bin/env python3
"""
main.py — Config-driven orchestrator for the full ConTopo pipeline.

The pipeline step graph is declared in conf/pipeline/*.yaml.
Hydra overrides are fully supported and forwarded to child processes.

Usage:
    python main.py                                  # full pipeline (default)
    python main.py pipeline=small                  # smoke-test pipeline
    python main.py pipeline=small pipeline.from_step=ensemble      # resume from ensemble step
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, cast


def run_script(script: str, overrides: list[str]) -> None:
    """Run a pipeline script as a subprocess."""
    script_path = os.path.join("scripts", script)
    cmd = [sys.executable, script_path] + overrides
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    steps = cast(
        list[dict[str, Any]],
        OmegaConf.to_container(OmegaConf.select(cfg, "pipeline.steps"), resolve=True),
    )
    from_step: str | None = OmegaConf.select(cfg, "pipeline.from_step")

    active = from_step is None
    for step in steps:
        step_id = step["id"]
        script = step["script"]
        description = step.get("description", "")
        sweep = step.get("sweep")
        extra_overrides: list[str] = step.get("overrides") or []

        if not active:
            if step_id == from_step:
                active = True
            else:
                print(f"[SKIP] {step_id}: {description}")
                continue

        overrides: list[str] = []
        if sweep:
            overrides.append(f"+sweeps={sweep}")

        # Normalize per-step overrides: each entry may be a string (shell-style)
        # or already a single override item. We must extend `overrides` with
        # individual string tokens so subprocess invocation receives a flat
        # list of strings.
        for o in extra_overrides:
            if isinstance(o, str):
                overrides.extend(shlex.split(o))
            else:
                # non-string entries (e.g. already a single override) are
                # appended as their string representation
                overrides.append(str(o))

        print(f"\n[STEP] {step_id}: {description}")
        run_script(script, overrides)

    print(f"\n{'='*70}")
    print("  Pipeline complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
