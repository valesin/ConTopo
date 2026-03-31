from __future__ import annotations

import torch

from src import mlflow_utils


def test_artifact_cache_hit_skips_download(tmp_path, monkeypatch):
    run_id = "run_hit"
    artifact_path = "inference_data/test_small.txt"
    cache_file = tmp_path / run_id / artifact_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("cached")

    called = {"n": 0}

    def _download_artifacts(*args, **kwargs):
        called["n"] += 1
        return "should_not_be_used"

    monkeypatch.setattr(
        mlflow_utils.mlflow.artifacts,
        "download_artifacts",
        _download_artifacts,
    )

    local_path = mlflow_utils.load_mlflow_artifact(
        run_id,
        artifact_path,
        file_type="auto",
        cache_dir=str(tmp_path),
    )

    assert local_path == str(cache_file)
    assert called["n"] == 0


def test_artifact_cache_miss_downloads_to_deterministic_path(tmp_path, monkeypatch):
    run_id = "run_miss"
    artifact_path = "inference_data/test_small.txt"
    expected_path = tmp_path / run_id / artifact_path

    calls = []

    def _download_artifacts(*args, **kwargs):
        calls.append(kwargs)
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text("downloaded")
        return str(expected_path)

    monkeypatch.setattr(
        mlflow_utils.mlflow.artifacts,
        "download_artifacts",
        _download_artifacts,
    )

    local_path = mlflow_utils.load_mlflow_artifact(
        run_id,
        artifact_path,
        file_type="auto",
        cache_dir=str(tmp_path),
    )

    assert local_path == str(expected_path)
    assert len(calls) == 1
    assert calls[0]["artifact_uri"] == f"runs:/{run_id}/{artifact_path}"
    assert calls[0]["dst_path"] == str(expected_path.parent)


def test_artifact_cache_refresh_forces_download(tmp_path, monkeypatch):
    run_id = "run_refresh"
    artifact_path = "inference_data/test_small.txt"
    expected_path = tmp_path / run_id / artifact_path
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text("old")

    calls = []

    def _download_artifacts(*args, **kwargs):
        calls.append(kwargs)
        expected_path.write_text("new")
        return str(expected_path)

    monkeypatch.setattr(
        mlflow_utils.mlflow.artifacts,
        "download_artifacts",
        _download_artifacts,
    )

    local_path = mlflow_utils.load_mlflow_artifact(
        run_id,
        artifact_path,
        file_type="auto",
        cache_dir=str(tmp_path),
        refresh_cache=True,
    )

    assert local_path == str(expected_path)
    assert expected_path.read_text() == "new"
    assert len(calls) == 1


def test_corrupt_torch_cache_retries_once(tmp_path, monkeypatch):
    run_id = "run_corrupt"
    artifact_path = "profiles/test_profiles.pt"
    cache_file = tmp_path / run_id / artifact_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("corrupt")

    calls = {"n": 0}

    def _download_artifacts(*args, **kwargs):
        calls["n"] += 1
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor([1.0, 2.0, 3.0]), cache_file)
        return str(cache_file)

    monkeypatch.setattr(
        mlflow_utils.mlflow.artifacts,
        "download_artifacts",
        _download_artifacts,
    )

    tensor = mlflow_utils.load_mlflow_artifact(
        run_id,
        artifact_path,
        file_type="torch",
        cache_dir=str(tmp_path),
    )

    assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))
    assert calls["n"] == 1


def test_unsafe_artifact_path_rejected(tmp_path):
    try:
        mlflow_utils.load_mlflow_artifact(
            "run_x",
            "../outside.txt",
            file_type="auto",
            cache_dir=str(tmp_path),
        )
        assert False, "Expected ValueError for unsafe artifact path"
    except ValueError as e:
        assert "Unsafe artifact_path" in str(e)


def test_cache_dir_required_when_cache_enabled():
    try:
        mlflow_utils.load_mlflow_artifact(
            "run_x",
            "inference_data/test_small.txt",
            file_type="auto",
            use_cache=True,
            cache_dir=None,
        )
        assert False, "Expected ValueError when cache_dir is missing"
    except ValueError as e:
        assert "cache_dir must be provided" in str(e)
