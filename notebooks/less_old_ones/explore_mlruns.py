import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import mlflow
    import json

    # Setup the environment strictly leveraging the provided utility
    # to fix Hydra pathing and tracking variables seamlessly.
    from src.config.notebook import setup_environment

    cfg, experiment = setup_environment(overrides=["mlflow.experiment_name=Morning"])
    return cfg, experiment, mlflow


@app.cell
def _(experiment, mlflow):
    filter_string_models = "tags.kind = 'model'"

    models = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string_models
    )

    models
    return (models,)


@app.cell
def _(experiment, mlflow):
    filter_string_inference = "tags.kind = 'inference'"

    inferences = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string_inference
    )

    inferences

    return (inferences,)


@app.cell
def _(inferences, models):
    import pandas as pd

    merged = pd.merge(
        inferences,
        models,
        left_on="params.trained_model_run_id",
        right_on="run_id",
        how="inner",  # Or 'left', 'outer' as needed
    )
    merged
    return


@app.cell
def _(experiment, mlflow):
    filter_string_profiles = "tags.kind = 'category_similarity_profile'"

    profiles = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string_profiles
    )

    profiles
    return


@app.cell
def _(cfg):
    from src.data.anchors import AnchorSpec, get_or_create_anchors
    from src.data.manifest import get_or_create_manifest
    from src.config.paths import get_cache_dir

    split = cfg.pipeline.split
    force = cfg.pipeline.force
    cache_dir = get_cache_dir(cfg)

    manifest = get_or_create_manifest(
        dataset_name=cfg.dataset.name,
        split=split,
        data_root=cfg.runtime.data_root,
        artifacts_root=str(cache_dir),
    )

    sel = cfg.pipeline.anchors
    anchor_spec = AnchorSpec(
        source_split=sel.source_split,
        per_class=sel.per_class,
        strategy=sel.strategy,
        order_by=sel.order_by,
        num_classes=cfg.dataset.num_classes,
    )

    anchors = get_or_create_anchors(
        manifest,
        spec=anchor_spec,
        artifacts_root=str(cache_dir),
    )
    a_spec_hash = anchors["spec_hash"]
    anchor_indices = anchors["anchor_indices"]

    print(
        f"Anchors identified. Total anchors: {len(anchor_indices)}. Spec Hash: {a_spec_hash}"
    )
    return


if __name__ == "__main__":
    app.run()
