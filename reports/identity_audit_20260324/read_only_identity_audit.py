#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.config.hash import identity_hash

TARGET_EXPERIMENT = "contopo"
OUTPUT_DIR = Path(
    "/home/vlr/Workspaces/Topographic/ConTopo/reports/identity_audit_20260324"
)


def flatten_identity_section_training(
    prefix: str, section: dict[str, Any]
) -> dict[str, str]:
    out: dict[str, str] = {}

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
            return
        if isinstance(node, list):
            out[path] = json.dumps(node, sort_keys=True)
            return
        out[path] = str(node)

    _walk(section, prefix)
    return out


def flatten_identity_section_migration(
    prefix: str, section: dict[str, Any]
) -> dict[str, str]:
    out: dict[str, str] = {}

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
            return
        if isinstance(node, (list, tuple)):
            out[path] = str(node)
            return
        out[path] = str(node)

    _walk(section, prefix)
    return out


def compute_model_identity_from_cfg_training(
    cfg: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    fields: dict[str, str] = {}
    fields.update(
        flatten_identity_section_training("model", cfg.get("model", {}) or {})
    )
    fields.update(flatten_identity_section_training("loss", cfg.get("loss", {}) or {}))
    fields.update(
        flatten_identity_section_training("dataset", cfg.get("dataset", {}) or {})
    )
    fields.update(
        flatten_identity_section_training("training", cfg.get("training", {}) or {})
    )

    model_hash = identity_hash(
        "model",
        schema_version=str(cfg.get("schema_version")),
        trial=str(cfg.get("trial")),
        seed=str(cfg.get("seed")),
        **fields,
    )
    return model_hash, fields


def compute_model_identity_from_cfg_migration(
    cfg: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    fields: dict[str, str] = {}
    fields.update(
        flatten_identity_section_migration("model", cfg.get("model", {}) or {})
    )
    fields.update(flatten_identity_section_migration("loss", cfg.get("loss", {}) or {}))
    fields.update(
        flatten_identity_section_migration("dataset", cfg.get("dataset", {}) or {})
    )
    fields.update(
        flatten_identity_section_migration("training", cfg.get("training", {}) or {})
    )

    model_hash = identity_hash(
        "model",
        schema_version=str(cfg.get("schema_version")),
        trial=str(cfg.get("trial")),
        seed=str(cfg.get("seed")),
        **fields,
    )
    return model_hash, fields


def discover_db_candidates() -> list[Path]:
    candidates: list[Path] = [
        Path("/home/vlr/Workspaces/Topographic/ConTopo/mlflow.db"),
        Path("/home/vlr/Workspaces/outputs/mlflow.db"),
    ]
    candidates.extend(
        sorted(Path("/home/vlr/.local/share/Trash/files").rglob("mlflow.db"))
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            unique.append(path)
    return unique


def sqlite_connect_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def db_diagnostics(db_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "db_path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": db_path.stat().st_size if db_path.exists() else None,
    }
    if not db_path.exists():
        return out

    try:
        con = sqlite_connect_ro(db_path)
        cur = con.cursor()
        cur.execute("PRAGMA integrity_check")
        out["integrity_check"] = cur.fetchone()[0]
        cur.execute("PRAGMA journal_mode")
        out["journal_mode"] = cur.fetchone()[0]
        cur.execute("SELECT version_num FROM alembic_version")
        row = cur.fetchone()
        out["alembic_version"] = row[0] if row else None
        cur.execute("SELECT COUNT(*) FROM runs")
        out["run_count"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tags")
        out["tag_count"] = cur.fetchone()[0]
        cur.execute(
            "SELECT experiment_id, name, artifact_location, lifecycle_stage FROM experiments ORDER BY experiment_id"
        )
        out["experiments"] = [
            {
                "experiment_id": r[0],
                "name": r[1],
                "artifact_location": r[2],
                "lifecycle_stage": r[3],
            }
            for r in cur.fetchall()
        ]
        con.close()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def get_model_runs_for_experiment(
    db_path: Path, experiment_name: str
) -> tuple[list[dict[str, Any]], str | None]:
    con = sqlite_connect_ro(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT experiment_id FROM experiments WHERE name = ? AND lifecycle_stage = 'active'",
        (experiment_name,),
    )
    row = cur.fetchone()
    if not row:
        con.close()
        return [], None
    experiment_id = str(row[0])

    cur.execute(
        """
        SELECT
            r.run_uuid,
            r.status,
            r.start_time,
            r.end_time,
            r.artifact_uri,
            r.experiment_id,
            MAX(CASE WHEN t.key='cfg_hash' THEN t.value END) AS cfg_hash,
            MAX(CASE WHEN t.key='identity_hash' THEN t.value END) AS identity_hash,
            MAX(CASE WHEN t.key='identity_hash_legacy' THEN t.value END) AS identity_hash_legacy,
            MAX(CASE WHEN t.key='kind' THEN t.value END) AS kind
        FROM runs r
        LEFT JOIN tags t ON r.run_uuid = t.run_uuid
        WHERE r.experiment_id = ?
          AND r.lifecycle_stage = 'active'
        GROUP BY r.run_uuid
        HAVING kind = 'model' AND status = 'FINISHED'
        ORDER BY r.start_time
        """,
        (experiment_id,),
    )
    rows = cur.fetchall()
    con.close()

    records: list[dict[str, Any]] = []
    for r in rows:
        records.append(
            {
                "run_id": r[0],
                "status": r[1],
                "start_time": r[2],
                "end_time": r[3],
                "artifact_uri": r[4],
                "experiment_id": r[5],
                "cfg_hash": r[6],
                "existing_identity_hash": r[7],
                "identity_hash_legacy": r[8],
                "kind": r[9],
            }
        )
    return records, experiment_id


def resolve_artifacts_root(db_path: Path, artifact_uri: str | None) -> Path | None:
    if artifact_uri:
        if artifact_uri.startswith("file://"):
            candidate = Path(artifact_uri.replace("file://", "", 1))
        else:
            candidate = Path(artifact_uri)
        if candidate.exists():
            return candidate

    if (
        db_path.parent.name.startswith("outputs")
        and (db_path.parent / "mlruns").exists()
    ):
        return db_path.parent / "mlruns"

    if (db_path.parent / "mlruns").exists():
        return db_path.parent / "mlruns"

    return None


def find_resolved_config_file(
    db_path: Path, run: dict[str, Any]
) -> tuple[Path | None, str]:
    run_id = run["run_id"]
    artifact_uri = run.get("artifact_uri")

    if artifact_uri:
        path = (
            Path(artifact_uri.replace("file://", "", 1))
            if str(artifact_uri).startswith("file://")
            else Path(str(artifact_uri))
        )
        if path.exists():
            config_dir = path / "config"
            if config_dir.exists():
                yamls = sorted(config_dir.glob("*.yaml"))
                if yamls:
                    return yamls[0], "artifact_uri_config_dir"

    roots: list[Path] = []
    root = resolve_artifacts_root(db_path, artifact_uri)
    if root:
        roots.append(root)

    # likely trash snapshot layout: outputs.X/mlruns/<run_id>/artifacts
    if db_path.parent.name.startswith("outputs"):
        roots.append(db_path.parent / "mlruns")

    # try in cwd local mlruns as fallback
    roots.append(Path("/home/vlr/Workspaces/Topographic/ConTopo/mlruns"))

    for artifacts_root in roots:
        run_artifacts = artifacts_root / run_id / "artifacts"
        config_dir = run_artifacts / "config"
        if config_dir.exists():
            yamls = sorted(config_dir.glob("*.yaml"))
            if yamls:
                return yamls[0], f"fallback_root:{artifacts_root}"

    return None, "config_missing"


def load_resolved_config(cfg_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        cfg = OmegaConf.load(str(cfg_path))
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(resolved, dict):
            return resolved, None
        return None, "resolved_config_not_dict"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def classify_run(db_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    cfg_file, cfg_file_source = find_resolved_config_file(db_path, run)
    result["config_file"] = str(cfg_file) if cfg_file else None
    result["config_file_source"] = cfg_file_source

    if cfg_file is None:
        result["computed_identity_hash"] = None
        result["computed_identity_hash_migration_helper"] = None
        result["classification"] = "missing_artifact"
        result["discrepancy_reason"] = "resolved_config_artifact_missing"
        result["differing_flattened_keys"] = []
        result["flattened_field_count"] = 0
        result["flattened_fields"] = {}
        result["flattened_fields_migration"] = {}
        return result

    cfg_dict, cfg_error = load_resolved_config(cfg_file)
    if cfg_error is not None or cfg_dict is None:
        result["computed_identity_hash"] = None
        result["computed_identity_hash_migration_helper"] = None
        result["classification"] = "needs_manual_inspection"
        result["discrepancy_reason"] = f"config_parse_failed:{cfg_error}"
        result["differing_flattened_keys"] = []
        result["flattened_field_count"] = 0
        result["flattened_fields"] = {}
        result["flattened_fields_migration"] = {}
        return result

    missing_required = [
        k
        for k in [
            "schema_version",
            "trial",
            "seed",
            "model",
            "loss",
            "dataset",
            "training",
        ]
        if k not in cfg_dict
    ]
    if missing_required:
        result["computed_identity_hash"] = None
        result["computed_identity_hash_migration_helper"] = None
        result["classification"] = "needs_manual_inspection"
        result["discrepancy_reason"] = (
            f"resolved_config_missing_keys:{','.join(missing_required)}"
        )
        result["differing_flattened_keys"] = []
        result["flattened_field_count"] = 0
        result["flattened_fields"] = {}
        result["flattened_fields_migration"] = {}
        return result

    try:
        computed, flattened = compute_model_identity_from_cfg_training(cfg_dict)
        computed_migration, flattened_migration = (
            compute_model_identity_from_cfg_migration(cfg_dict)
        )
    except Exception as exc:
        result["computed_identity_hash"] = None
        result["computed_identity_hash_migration_helper"] = None
        result["classification"] = "needs_manual_inspection"
        result["discrepancy_reason"] = (
            f"identity_compute_failed:{type(exc).__name__}:{exc}"
        )
        result["differing_flattened_keys"] = []
        result["flattened_field_count"] = 0
        result["flattened_fields"] = {}
        result["flattened_fields_migration"] = {}
        return result

    result["computed_identity_hash"] = computed
    result["computed_identity_hash_migration_helper"] = computed_migration
    result["flattened_field_count"] = len(flattened)
    result["flattened_fields"] = flattened
    result["flattened_fields_migration"] = flattened_migration

    diff_keys = sorted(
        k
        for k in set(flattened.keys()) | set(flattened_migration.keys())
        if flattened.get(k) != flattened_migration.get(k)
    )
    result["differing_flattened_keys"] = diff_keys

    existing = run.get("existing_identity_hash")
    if existing is None:
        result["classification"] = "needs_backfill"
        if computed == computed_migration:
            result["discrepancy_reason"] = "missing_identity_hash"
        else:
            result["discrepancy_reason"] = (
                "missing_identity_hash_and_training_vs_migration_hash_diff"
            )
    elif existing == computed:
        result["classification"] = "ok"
        result["discrepancy_reason"] = "existing_matches_training_computation"
    elif existing == computed_migration:
        result["classification"] = "needs_manual_inspection"
        result["discrepancy_reason"] = "existing_matches_migration_helper_not_training"
    else:
        result["classification"] = "needs_manual_inspection"
        result["discrepancy_reason"] = "existing_mismatch_unknown"

    return result


def grouped_presence_summary(
    records: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get(key))].append(rec)

    out: list[dict[str, Any]] = []
    for group_key, rows in sorted(groups.items(), key=lambda x: x[0]):
        with_identity = sum(1 for row in rows if row.get("existing_identity_hash"))
        without_identity = sum(
            1 for row in rows if not row.get("existing_identity_hash")
        )
        out.append(
            {
                "group_key": group_key,
                "count": len(rows),
                "with_identity": with_identity,
                "without_identity": without_identity,
                "mixed_presence": with_identity > 0 and without_identity > 0,
                "run_ids": [row["run_id"] for row in rows],
            }
        )
    return out


def run_audit() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    db_candidates = discover_db_candidates()
    db_diags = [db_diagnostics(db_path) for db_path in db_candidates]

    all_records: list[dict[str, Any]] = []
    per_db_summary: list[dict[str, Any]] = []

    for db_path in db_candidates:
        model_runs, experiment_id = get_model_runs_for_experiment(
            db_path, TARGET_EXPERIMENT
        )
        if not model_runs:
            per_db_summary.append(
                {
                    "db_path": str(db_path),
                    "target_experiment": TARGET_EXPERIMENT,
                    "experiment_id": experiment_id,
                    "model_run_count": 0,
                    "note": "no_finished_model_runs",
                }
            )
            continue

        classified = [classify_run(db_path, run) for run in model_runs]
        all_records.extend(
            {
                "db_path": str(db_path),
                "target_experiment": TARGET_EXPERIMENT,
                **record,
            }
            for record in classified
        )

        cls_counts = Counter(r["classification"] for r in classified)
        per_db_summary.append(
            {
                "db_path": str(db_path),
                "target_experiment": TARGET_EXPERIMENT,
                "experiment_id": experiment_id,
                "model_run_count": len(classified),
                "class_counts": dict(cls_counts),
            }
        )

    cfg_group_summary = (
        grouped_presence_summary(all_records, "cfg_hash") if all_records else []
    )
    id_group_summary = (
        grouped_presence_summary(all_records, "existing_identity_hash")
        if all_records
        else []
    )

    # variability analysis of flattened keys within cfg_hash groups
    variability: dict[str, dict[str, list[str]]] = {}
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_records:
        by_cfg[str(rec.get("cfg_hash"))].append(rec)

    for cfg_key, records in by_cfg.items():
        key_values: dict[str, set[str]] = defaultdict(set)
        for rec in records:
            for field_key, field_val in (rec.get("flattened_fields") or {}).items():
                key_values[field_key].add(str(field_val))
        varying = {k: sorted(v) for k, v in key_values.items() if len(v) > 1}
        if varying:
            variability[cfg_key] = varying

    candidate_collisions: list[dict[str, Any]] = []
    by_computed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_records:
        comp = rec.get("computed_identity_hash")
        if comp:
            by_computed[str(comp)].append(rec)
    for comp, rows in by_computed.items():
        if len(rows) > 1:
            cfgs = sorted(set(str(r.get("cfg_hash")) for r in rows))
            run_ids = [r["run_id"] for r in rows]
            candidate_collisions.append(
                {
                    "computed_identity_hash": comp,
                    "count": len(rows),
                    "cfg_hashes": cfgs,
                    "run_ids": run_ids,
                    "possible_duplicate_semantics": len(cfgs) > 1,
                }
            )

    report = {
        "target_experiment": TARGET_EXPERIMENT,
        "db_candidates": [str(p) for p in db_candidates],
        "db_diagnostics": db_diags,
        "per_db_summary": per_db_summary,
        "run_records": all_records,
        "group_by_cfg_hash": cfg_group_summary,
        "group_by_existing_identity_hash": id_group_summary,
        "flattened_variability_by_cfg_hash": variability,
        "candidate_identity_collisions": candidate_collisions,
    }

    json_path = OUTPUT_DIR / "identity_audit_contopo.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    csv_path = OUTPUT_DIR / "identity_audit_contopo.csv"
    csv_fields = [
        "db_path",
        "target_experiment",
        "run_id",
        "cfg_hash",
        "existing_identity_hash",
        "computed_identity_hash",
        "computed_identity_hash_migration_helper",
        "classification",
        "discrepancy_reason",
        "status",
        "start_time",
        "end_time",
        "config_file",
        "config_file_source",
        "identity_hash_legacy",
        "flattened_field_count",
        "differing_flattened_keys",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_records:
            flat = {k: row.get(k) for k in csv_fields}
            flat["differing_flattened_keys"] = json.dumps(
                row.get("differing_flattened_keys", []), ensure_ascii=False
            )
            writer.writerow(flat)

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "record_count": len(all_records),
        "db_count": len(db_candidates),
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2))
