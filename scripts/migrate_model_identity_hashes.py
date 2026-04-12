#!/usr/bin/env python3
"""Backfill identity_hash tags on existing model runs after a schema change.

Each time the model identity hash schema changes (fields added, removed, or
values renamed), existing MLflow runs carry stale hashes and are no longer
recognised as idempotent. This script recomputes the correct hash for every
FINISHED model run using the run's stored config YAML artifact, then writes
the updated tag back to MLflow.

See docs/identity_hash_migration.md for the full decision guide.

── HOW IT WORKS ────────────────────────────────────────────────────────────
For each run the script:
  1. Downloads the resolved config YAML that was logged at training time.
  2. Filters each config section through the *current* structured schema
     (strips removed fields, fills missing ones with current defaults,
     coerces types) — so the result mirrors what 01_train_models.py sees.
  3. Applies value-rename aliases (e.g. strategy renames) via _normalise_*.
  4. Computes identity_hash("model", **fields) and compares with stored tag.
  5. Writes the new tag if they differ (only with --apply).

── CUSTOMISATION POINTS ────────────────────────────────────────────────────
When you make a schema change, edit ONE OR BOTH of the following:

  A. Field added or removed from structured config (src/config/structured.py)
     → No code change needed here. _canonical_section() handles this
       automatically: removed fields are stripped, added fields get their
       current default value. Verify with a dry-run.

  B. Field VALUE renamed (e.g. a strategy string changed name)
     → Add an entry to the relevant _*_ALIASES dict below, and make sure
       _normalise_*() applies it before _canonical_section() is called.
       Pattern: {"old_value": "new_value"}.

── USAGE ────────────────────────────────────────────────────────────────────
    # Preview changes (no writes):
    uv run scripts/migrate_model_identity_hashes.py --experiment contopo

    # Preview with field dump for the first patched run:
    uv run scripts/migrate_model_identity_hashes.py --experiment contopo --verbose

    # Apply:
    uv run scripts/migrate_model_identity_hashes.py --experiment contopo --apply
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict

from mlflow.tracking import MlflowClient
import mlflow
from omegaconf import OmegaConf
from src.repositories.functional_run_repository import (
    configure_run_repository,
    search_runs,
)

from src.config.hash import identity_hash
from src.config.structured import DatasetConfig, LossConfig, ModelConfig, TrainingConfig


def _canonical_section(stored: dict, struct_class) -> dict:
    """Produce a canonical section dict that exactly matches what the training script sees.

    Strategy:
      1. Start with the current structured config (provides all current defaults).
      2. Strip unknown keys from `stored` (fields removed from the schema).
      3. Merge filtered stored values into the struct (provides missing-field defaults
         and type-coerces values to declared types, e.g. int 0 -> float 0.0).

    This is equivalent to what OmegaConf.to_container(cfg.section, resolve=True)
    produces in the training script.
    """
    struct_node = OmegaConf.structured(struct_class)
    template = OmegaConf.to_container(struct_node)

    def _filter(src: dict, tmpl: dict) -> dict:
        result = {}
        for k, v in src.items():
            if k not in tmpl:
                continue
            if isinstance(v, dict) and isinstance(tmpl[k], dict):
                result[k] = _filter(v, tmpl[k])
            else:
                result[k] = v
        return result

    filtered = _filter(stored, template)
    merged = OmegaConf.merge(struct_node, OmegaConf.create(filtered))
    return OmegaConf.to_container(merged, resolve=True)


def _flatten_section(prefix: str, section: Dict) -> Dict[str, str]:
    """Flatten a config section to dot-path string fields.

    Must exactly mirror _flatten_identity_section in scripts/01_train_models.py.
    Lists are serialised with json.dumps (not str()) to match that function.
    """
    out: Dict[str, str] = {}

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}")
            return
        if isinstance(node, list):
            out[path] = json.dumps(node, sort_keys=True)
            return
        out[path] = str(node)

    _walk(section, prefix)
    return out


def find_resolved_config_artifact(client: MlflowClient, run_id: str) -> str | None:
    """Return the artifact path of the stored config YAML, or None."""
    artifacts = client.list_artifacts(run_id, path="config")
    for art in artifacts:
        if art.path and art.path.endswith(".yaml"):
            return art.path
    return None


def load_resolved_cfg(run_id: str, artifact_relpath: str) -> dict | None:
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_relpath}"
        )
        cfg = OmegaConf.load(local_path)
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None


# Maps old config values to their current equivalents, applied before hashing.
_STRATEGY_ALIASES: Dict[str, str] = {
    "seeded_per_class": "first_n_per_class",
}


def _normalise_dataset(dataset: dict) -> dict:
    """Apply known value renames so old stored configs hash identically to new ones."""
    dataset = dict(dataset)
    split = dict(dataset.get("split", {}))
    if split.get("strategy") in _STRATEGY_ALIASES:
        split["strategy"] = _STRATEGY_ALIASES[split["strategy"]]
    dataset["split"] = split
    return dataset


def compute_model_identity_from_cfg(cfg: dict) -> tuple[str, Dict[str, str]]:
    """Compute the current-schema identity hash from a resolved config dict.

    Returns (hash, identity_fields) so callers can inspect what was hashed.

    Mirrors model_identity_fields + identity_hash("model") in src/config/hash.py.
    Filters each section through the current structured config schema so that
    fields removed at any point (e.g. dataset.num_classes, dataset.split.seed)
    are excluded regardless of what the stored YAML contains.
    """
    schema_version = str(cfg.get("schema_version"))
    trial = str(cfg.get("trial"))
    seed = str(cfg.get("seed"))

    model_section = _canonical_section(cfg.get("model", {}), ModelConfig)
    loss_section = _canonical_section(cfg.get("loss", {}), LossConfig)
    dataset_section = _canonical_section(
        _normalise_dataset(cfg.get("dataset", {})), DatasetConfig
    )
    training_section = _canonical_section(cfg.get("training", {}), TrainingConfig)

    fields: Dict[str, str] = {}
    fields.update(_flatten_section("model", model_section))
    fields.update(_flatten_section("loss", loss_section))
    fields.update(_flatten_section("dataset", dataset_section))
    fields.update(_flatten_section("training", training_section))

    all_fields = {
        "schema_version": schema_version,
        "trial": trial,
        "seed": seed,
        **fields,
    }
    h = identity_hash("model", **all_fields)
    return h, all_fields


def main():
    parser = argparse.ArgumentParser(
        description="Backfill model identity_hash tags after schema changes."
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument(
        "--apply", action="store_true", help="Actually write tags (default: dry-run)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap number of runs processed (0=all)"
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print computed identity fields for each patched run",
    )
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()
    configure_run_repository(mlflow.get_tracking_uri(), args.experiment)

    if not args.apply:
        print("DRY RUN — pass --apply to write tags.\n")

    filter_str = "tags.kind = 'model' and attributes.status = 'FINISHED'"
    runs = search_runs(filter_str, output_format="pandas")
    print(f"Found {len(runs)} FINISHED model runs.")

    processed = 0
    updated = 0
    skipped = 0

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        existing_identity = client.get_run(run_id).data.tags.get("identity_hash")

        artifact_rel = find_resolved_config_artifact(client, run_id)
        if artifact_rel is None:
            logging.warning("No config artifact for run %s; skipping", run_id)
            skipped += 1
            continue

        cfg = load_resolved_cfg(run_id, artifact_rel)
        if cfg is None:
            logging.warning("Failed loading config for run %s; skipping", run_id)
            skipped += 1
            continue

        try:
            new_identity, identity_fields = compute_model_identity_from_cfg(cfg)
        except Exception as e:
            logging.warning(
                "Hash computation failed for run %s: %s; skipping", run_id, e
            )
            skipped += 1
            continue

        if existing_identity == new_identity:
            print(f"OK    {run_id}")
        else:
            print(f"PATCH {run_id}  {existing_identity or '(none)'} -> {new_identity}")
            if args.verbose:
                print(json.dumps(dict(sorted(identity_fields.items())), indent=4))
            if args.apply:
                client.set_tag(run_id, "identity_hash", new_identity)
            updated += 1

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"\nDone. processed={processed}, patched={updated}, skipped={skipped}")
    if not args.apply and updated > 0:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
