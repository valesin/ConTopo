#!/usr/bin/env python3
"""Recompute identity_hash tags on existing model runs after a TrainingConfig schema change.

When new fields are added to TrainingConfig (which is covered by the `training.*`
wildcard in the model identity hash), every existing model run has a stale
`identity_hash` tag. This script recomputes the correct hash for every FINISHED
model run and writes the updated tag back to MLflow.

How it works:
  For each FINISHED model run the script:
    1. Downloads the resolved config YAML that was logged at training time
       (artifact path: config/resolved_config.yaml).
    2. Filters each config section through the *current* structured schema via
       _canonical_section() — strips removed fields, fills missing ones with the
       current default values (e.g. new TrainingConfig fields get their migration
       defaults automatically).
    3. Computes identity_hash("model", **model_identity_fields(reconstructed_cfg, seed))
       and compares with the stored tag.
    4. Writes the new tag if they differ (only with --apply).

Customisation points:
  A. Field added or removed from structured config (src/config/structured.py)
     → No code change needed here. _canonical_section() handles this automatically:
       removed fields are stripped, added fields get the current default value.
       Verify with a dry-run.

  B. Field VALUE renamed (e.g. a strategy string changed name)
     → Add an entry to the relevant *_ALIASES dict below and make sure the
       corresponding _normalise_*() function applies it before _canonical_section()
       is called. Pattern: {"old_value": "new_value"}.

── USAGE ────────────────────────────────────────────────────────────────────
    # Preview changes (no writes):
    uv run scripts/migrations/rehash_identities.py --experiment contopo

    # Preview with field dump for the first mismatched run:
    uv run scripts/migrations/rehash_identities.py --experiment contopo --verbose

    # Apply:
    uv run scripts/migrations/rehash_identities.py --experiment contopo --apply

    # Limit to first N runs (useful for spot-checking):
    uv run scripts/migrations/rehash_identities.py --experiment contopo --limit 5

    # Custom tracking URI:
    uv run scripts/migrations/rehash_identities.py --experiment contopo \\
        --tracking-uri sqlite:///outputs/mlflow.db

── RUNNING ORDER ────────────────────────────────────────────────────────────
    Run AFTER backfill_params.py (param backfill first, then rehash).
    See CONTRIBUTING_AND_UPDATING.md §11 for the full migration protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from typing import Dict

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.repositories.functional_run_repository import (
    configure_run_repository,
    search_runs,
)
from src.config.hash import identity_hash, model_identity_fields
from src.config.structured import DatasetConfig, LossConfig, ModelConfig, TrainingConfig

log = logging.getLogger(__name__)


# ── Value-rename aliases ─────────────────────────────────────────────────────
# Add entries here if a field *value* was renamed (not the field itself).
# Example: {"old_strategy_name": "new_strategy_name"}
_TRAINING_ALIASES: Dict[str, str] = {}
_DATASET_ALIASES: Dict[str, str] = {}
_MODEL_ALIASES: Dict[str, str] = {}
_LOSS_ALIASES: Dict[str, str] = {}


def _normalise_training(d: dict) -> dict:
    d = dict(d)
    if "optimiser" in d and d["optimiser"] in _TRAINING_ALIASES:
        d["optimiser"] = _TRAINING_ALIASES[d["optimiser"]]
    return d


def _normalise_dataset(d: dict) -> dict:
    return d


def _normalise_model(d: dict) -> dict:
    return d


def _normalise_loss(d: dict) -> dict:
    return d


# ── Canonical section helper ─────────────────────────────────────────────────

def _canonical_section(stored: dict, struct_class) -> dict:
    """Produce a canonical section dict that exactly matches what the training
    script sees after OmegaConf composition.

    Strategy:
      1. Start with the current structured config (provides all current defaults).
      2. Strip unknown keys from ``stored`` (fields removed from schema).
      3. Merge filtered stored values into the struct (type-coerces and fills
         missing fields with current defaults — e.g. new TrainingConfig fields
         get their migration default automatically).

    This mirrors OmegaConf.to_container(cfg.section, resolve=True) in the script.
    """
    struct_node = OmegaConf.structured(struct_class)
    template = OmegaConf.to_container(struct_node)

    def _filter(src: dict, tmpl: dict) -> dict:
        result = {}
        for k, v in src.items():
            if k not in tmpl:
                continue  # field removed from schema — drop it
            if isinstance(v, dict) and isinstance(tmpl[k], dict):
                result[k] = _filter(v, tmpl[k])
            else:
                result[k] = v
        return result

    filtered = _filter(stored, template)
    merged = OmegaConf.merge(struct_node, OmegaConf.create(filtered))
    return OmegaConf.to_container(merged, resolve=True)


def _flatten_section(prefix: str, section: dict) -> Dict[str, str]:
    """Flatten a config section to dot-path string fields.

    Must exactly mirror flatten_identity_section in src/config/hash.py.
    Lists are serialised with json.dumps (not str()) to match that function.
    """
    out: Dict[str, str] = {}

    def _walk(node, path: str):
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


def _recompute_identity_hash(run_id: str, client: MlflowClient, verbose: bool) -> str | None:
    """Download stored config, reconstruct identity fields, return new hash.

    Returns None if the config artifact cannot be loaded.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try the canonical path first (runs logged after the naming fix).
        # Fall back to listing config/ and taking the first .yaml found
        # (runs logged before the fix used a random tempfile name).
        local_path = None
        try:
            local_path = client.download_artifacts(run_id, "config/resolved_config.yaml", tmpdir)
        except Exception:
            pass

        if local_path is None:
            try:
                artifacts = client.list_artifacts(run_id, "config")
                yaml_artifacts = [a for a in artifacts if a.path.endswith(".yaml")]
                if not yaml_artifacts:
                    log.warning("run %s — no YAML artifact found under config/", run_id)
                    return None
                local_path = client.download_artifacts(run_id, yaml_artifacts[0].path, tmpdir)
            except Exception as exc:
                log.warning("run %s — could not download config artifact: %s", run_id, exc)
                return None

        raw = OmegaConf.load(local_path)
        stored = OmegaConf.to_container(raw, resolve=True)

    # Extract stored sections
    stored_model    = _normalise_model(stored.get("model", {}))
    stored_loss     = _normalise_loss(stored.get("loss", {}))
    stored_dataset  = _normalise_dataset(stored.get("dataset", {}))
    stored_training = _normalise_training(stored.get("training", {}))

    # Canonicalise against current structured schema
    canon_model    = _canonical_section(stored_model, ModelConfig)
    canon_loss     = _canonical_section(stored_loss, LossConfig)
    canon_dataset  = _canonical_section(stored_dataset, DatasetConfig)
    canon_training = _canonical_section(stored_training, TrainingConfig)

    # Reconstruct identity fields
    schema_version = str(stored.get("schema_version", "1"))
    trial          = str(stored.get("trial", "0"))
    seed           = str(stored.get("seed", "None"))

    fields: Dict[str, str] = {
        "schema_version": schema_version,
        "trial": trial,
        "seed": seed,
    }
    fields.update(_flatten_section("model", canon_model))
    fields.update(_flatten_section("loss", canon_loss))
    fields.update(_flatten_section("dataset", canon_dataset))
    fields.update(_flatten_section("training", canon_training))

    if verbose:
        print("  Fields:")
        for k, v in sorted(fields.items()):
            print(f"    {k} = {v!r}")

    return identity_hash("model", **fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute identity_hash tags on FINISHED model runs after a schema change."
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write updated tags (default: dry-run)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print reconstructed identity fields for each changed run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap number of runs processed (0 = all)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///outputs/mlflow.db",
        help="MLflow tracking URI (default: sqlite:///outputs/mlflow.db)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    configure_run_repository(args.tracking_uri, args.experiment)
    client = MlflowClient()

    if not args.apply:
        print("DRY RUN — pass --apply to write tags.\n")

    filter_str = "tags.kind = 'model' and attributes.status = 'FINISHED'"
    runs = search_runs(filter_str, output_format="pandas")
    print(f"Found {len(runs)} FINISHED model runs.")

    processed = 0
    patched = 0
    skipped = 0
    errors = 0

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        stored_hash = client.get_run(run_id).data.tags.get("identity_hash", "")

        new_hash = _recompute_identity_hash(run_id, client, verbose=args.verbose)
        if new_hash is None:
            print(f"ERROR {run_id}  (could not load config artifact)")
            errors += 1
        elif new_hash == stored_hash:
            print(f"SKIP  {run_id}  (hash unchanged: {stored_hash})")
            skipped += 1
        else:
            status = "PATCH" if args.apply else "WOULD PATCH"
            print(
                f"{status} {run_id}  "
                f"old={stored_hash!r}  →  new={new_hash!r}"
            )
            if args.apply:
                client.set_tag(run_id, "identity_hash", new_hash)
            patched += 1

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(
        f"\nDone. runs_processed={processed}, "
        f"patched={patched}, skipped={skipped}, errors={errors}"
    )
    if not args.apply and patched > 0:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
