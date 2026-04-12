from __future__ import annotations

from typing import NamedTuple
import torch
import numpy as np

from omegaconf import DictConfig
from src.repositories.functional_run_repository import find_finished_identity_run
from src.mlflow_utils import load_mlflow_artifact
from src.config.hash import identity_hash
from src.profiling.masking import assert_valid_feature_tensor

class ComponentFeatures(NamedTuple):
    base_tensors: list[torch.Tensor]
    profile_tensors: list[torch.Tensor]
    component_logit_preds: list[torch.Tensor | None]

def extract_component_features(
    run_ids: list[str],
    split: str,
    feature_type: str,
    use_profiles: bool,
    similarity_metric: str,
    anchor_spec_hash: str,
    cfg: DictConfig,
    total_examples: int,
) -> ComponentFeatures:
    """Extract component inference and profiling features across the provided ml-runs."""
    base_tensors = []
    profile_tensors = []
    component_logit_preds = []

    for run_id in run_ids:
        inf_identity = identity_hash(
            "inference", trained_model_run_id=run_id, split=split
        )
        inf_run = find_finished_identity_run("inference", inf_identity)
        if inf_run is None:
            raise RuntimeError(f"Missing '{split}' inference for {run_id}")
        inf_run_id = inf_run.info.run_id

        if use_profiles:
            prof_identity = identity_hash(
                "category_similarity_profile",
                parent_run_id=run_id,
                anchor_spec_hash=anchor_spec_hash,
                similarity_metric=similarity_metric,
                split=split,
            )
            prof_run = find_finished_identity_run(
                "category_similarity_profile", prof_identity
            )
            if prof_run is None:
                raise RuntimeError(
                    f"Missing '{similarity_metric}' profile for {run_id}"
                )
            prof_run_id = prof_run.info.run_id
            inf_prof_node = load_mlflow_artifact(
                prof_run_id,
                f"profiles/{split}_{similarity_metric}_profiles.pt",
                file_type="torch",
                strict=True,
                cache_dir=cfg.mlflow.artifact_cache_dir,
            ).cpu()

            if not torch.isfinite(inf_prof_node).all():
                raise ValueError(
                    f"Profile artifact for model {run_id} (profile run {prof_run_id}) "
                    f"contains NaN/Inf — likely corrupted artifact on disk"
                )

            profile_tensors.append(inf_prof_node)

        data = load_mlflow_artifact(
            inf_run_id,
            f"inference/{split}_tensors.npz",
            file_type="numpy",
            strict=True,
            cache_dir=cfg.mlflow.artifact_cache_dir,
        )

        if "logits" in data:
            component_logit_preds.append(torch.from_numpy(data["logits"]))
        else:
            component_logit_preds.append(None)

        if "logits" in feature_type:
            if "logits" not in data:
                raise KeyError(
                    f"Missing logits in inference/{split}_tensors.npz for run {inf_run_id}"
                )
            base_tensor = torch.from_numpy(data["logits"])
        elif "embeddings" in feature_type:
            if "embeddings" not in data:
                raise KeyError(
                    f"Missing embeddings in inference/{split}_tensors.npz for run {inf_run_id}"
                )
            base_tensor = torch.nn.functional.normalize(
                torch.from_numpy(data["embeddings"]).float(), p=2, dim=1
            )
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        assert_valid_feature_tensor(
            "base_tensor", base_tensor, total_examples
        )
        base_tensors.append(base_tensor)

    return ComponentFeatures(
        base_tensors=base_tensors,
        profile_tensors=profile_tensors,
        component_logit_preds=component_logit_preds,
    )
