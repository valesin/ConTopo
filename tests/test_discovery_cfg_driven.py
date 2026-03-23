"""
Tests that discover_ensembles_from_cfg correctly reads cfg.groups.*
and delegates to the internal _discover function.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.ensemble.selector import discover_ensembles_from_cfg, _discover


def _make_cfg(group_by=None, min_components=2, filter_dict=None):
    return OmegaConf.create(
        {
            "groups": {
                "group_by": group_by or ["topology", "rho"],
                "min_components": min_components,
                "filter": filter_dict or {},
            }
        }
    )


def _fake_run(run_id: str, params: dict):
    r = MagicMock()
    r.info.run_id = run_id
    r.data.params = params
    return r


@patch("src.ensemble.selector.mlflow")
def test_cfg_params_forwarded_to_discover(mock_mlflow):
    """discover_ensembles_from_cfg passes cfg.groups.* to _discover correctly."""
    exp = MagicMock()
    exp.experiment_id = "exp1"
    mock_mlflow.get_experiment_by_name.return_value = exp

    runs = [
        _fake_run("r1", {"topology": "torus", "rho": "0.05"}),
        _fake_run("r2", {"topology": "torus", "rho": "0.05"}),
    ]
    mock_mlflow.search_runs.return_value = runs

    cfg = _make_cfg(group_by=["topology", "rho"], min_components=2)
    result = discover_ensembles_from_cfg(cfg, "my_experiment")

    assert len(result) == 1
    key = list(result.keys())[0]
    assert "torus" in key and "0.05" in key
    assert set(result[key]) == {"r1", "r2"}


@patch("src.ensemble.selector.mlflow")
def test_min_components_filters_small_groups(mock_mlflow):
    """Groups with fewer runs than min_components are excluded."""
    exp = MagicMock()
    exp.experiment_id = "exp1"
    mock_mlflow.get_experiment_by_name.return_value = exp

    runs = [
        _fake_run("r1", {"topology": "torus", "rho": "0.05"}),
        _fake_run("r2", {"topology": "torus", "rho": "0.05"}),
        _fake_run("r3", {"topology": "grid", "rho": "0.05"}),  # only 1 — below min
    ]
    mock_mlflow.search_runs.return_value = runs

    cfg = _make_cfg(group_by=["topology", "rho"], min_components=2)
    result = discover_ensembles_from_cfg(cfg, "my_experiment")

    assert len(result) == 1
    assert all("grid" not in k for k in result)


@patch("src.ensemble.selector.mlflow")
def test_cfg_driven_matches_direct_discover(mock_mlflow):
    """discover_ensembles_from_cfg and _discover return identical results."""
    exp = MagicMock()
    exp.experiment_id = "exp1"
    mock_mlflow.get_experiment_by_name.return_value = exp

    runs = [
        _fake_run("r1", {"topology": "torus", "rho": "0.05"}),
        _fake_run("r2", {"topology": "torus", "rho": "0.05"}),
        _fake_run("r3", {"topology": "grid", "rho": "0.2"}),
        _fake_run("r4", {"topology": "grid", "rho": "0.2"}),
    ]
    # Return same runs for both calls
    mock_mlflow.search_runs.return_value = runs

    cfg = _make_cfg(group_by=["topology", "rho"], min_components=2)
    result_cfg = discover_ensembles_from_cfg(cfg, "exp")

    mock_mlflow.search_runs.return_value = runs
    result_direct = _discover(
        experiment_name="exp",
        group_by=["topology", "rho"],
        min_components=2,
        base_filter={},
    )

    assert result_cfg == result_direct
