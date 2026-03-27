"""
Migrate MLflow artifact paths in the SQLite DB.

Updates all artifact URIs from the old container path to the current working directory.
Usage: python scripts/migrate_mlflow_paths.py [--dry-run]
"""

import argparse
import sqlite3
from pathlib import Path

OLD_ROOT = "/persistent_repo/ConTopo"


def migrate(db_path: Path, new_root: str, dry_run: bool) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = {
        "runs": "artifact_uri",
        "experiments": "artifact_location",
    }

    for table, column in tables.items():
        cur.execute(f"SELECT run_uuid, {column} FROM {table} WHERE {column} LIKE ?", (f"%{OLD_ROOT}%",))  # noqa: S608
        rows = cur.fetchall()

        if not rows:
            print(f"{table}.{column}: nothing to update")
            continue

        print(f"\n{table}.{column}: {len(rows)} rows to update")
        for row_id, old_val in rows:
            new_val = old_val.replace(OLD_ROOT, new_root)
            print(f"  {old_val!r}\n  -> {new_val!r}")
            if not dry_run:
                cur.execute(f"UPDATE {table} SET {column} = ? WHERE run_uuid = ?", (new_val, row_id))  # noqa: S608

    if dry_run:
        print("\n[dry-run] No changes written.")
    else:
        con.commit()
        print("\nDone. Changes committed.")

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "outputs" / "mlflow.db"
    new_root = str(project_root)

    print(f"DB:       {db_path}")
    print(f"Old root: {OLD_ROOT}")
    print(f"New root: {new_root}")

    if not db_path.exists():
        raise FileNotFoundError(f"MLflow DB not found at {db_path}")

    migrate(db_path, new_root, dry_run=args.dry_run)
