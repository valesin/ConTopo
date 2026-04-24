"""
Structured (dataclass) configs for Hydra.

Each dataclass mirrors one YAML config group under ``conf/``.
They are registered in the ConfigStore so Hydra validates config files
against the schema at composition time.

Usage::

    from src.config.structured import register_configs
    register_configs()  # call once, before @hydra.main
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

# ─────────── model ───────────


@dataclass
class HeadConfig:
    bias: bool = True


@dataclass
class ModelConfig:
    name: str = "resnet18"
    arch: str = "LinearResNet18"
    embedding_dim: int = 256
    use_dropout: bool = True
    p_dropout: float = 0.5
    head: HeadConfig = field(default_factory=HeadConfig)


# ─────────── loss ───────────


@dataclass
class NeighbourhoodConfig:
    type: str = "moore"
    radius: int = 1


@dataclass
class LossConfig:
    name: str = "cross_entropy"
    rho: float = 0.0
    topography_type: str = "ws"
    topology: str = "torus"
    task_loss: str = "cross_entropy"
    neighbourhood: NeighbourhoodConfig = field(default_factory=NeighbourhoodConfig)


# ─────────── dataset ───────────


@dataclass
class SplitConfig:
    strategy: str = "first_n_per_class"
    val_per_class: int = 500


@dataclass
class TransformsConfig:
    preset: str = "cifar10_resizedcrop_v1"


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    image_size: int = 32
    in_channels: int = 3
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    split: SplitConfig = field(default_factory=SplitConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)


# ─────────── training ───────────


@dataclass
class BalancerConfig:
    beta: float = 0.1
    eps: float = 1e-8
    lambda_max: float = 10000.0


@dataclass
class TrainingBetonConfig:
    """FFCV beton data format settings. Hash-included via training.*.
    All fields are None unless loading_backend=ffcv."""

    max_resolution: Optional[int] = None
    jpeg_quality: Optional[int] = None
    compress_probability: Optional[float] = None


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 0.002
    optimiser: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    scheduler: str = "none"  # "none" | "cyclic" | "cosine"
    lr_peak_epoch: Optional[int] = (
        None  # only for scheduler=cyclic; None means not applicable
    )
    amp: bool = False
    save_freq_epochs: int = 20
    save_checkpoints: bool = False
    early_stopping_patience: int = 25
    early_stopping_method: str = "val_acc"
    label_smoothing: float = 0.0  # CrossEntropyLoss label smoothing
    use_blurpool: bool = False  # antialiased pooling on strided layers
    optimizer_selective_wd: bool = False  # WD only on non-BN/bias params
    lr_tta: bool = False  # test-time augmentation (LR flip average)
    progressive_res_min: Optional[int] = None  # start resolution (progressive training)
    progressive_res_max: Optional[int] = None  # end resolution
    progressive_res_start_ramp: Optional[float] = (
        None  # only when progressive_res_min is set
    )
    progressive_res_end_ramp: Optional[float] = (
        None  # only when progressive_res_min is set
    )
    loading_backend: str = "torch"  # "torch" | "ffcv" — hash-included
    beton: TrainingBetonConfig = field(default_factory=TrainingBetonConfig)
    balancer: BalancerConfig = field(default_factory=BalancerConfig)


# ─────────── runtime ───────────


@dataclass
class InferenceRuntimeConfig:
    batch_size: int = 256
    num_workers: int = 2


@dataclass
class PathsConfig:
    """Centralized output paths, all relative to outputs_root."""

    anchors: str = "anchors"


@dataclass
class BetonConfig:
    """FFCV .beton file storage location (hash-excluded — operational only)."""

    dir: str = "outputs/betons"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    data_parallel: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    print_freq: int = 10
    verbose_upload_log: bool = False
    data_root: str = "./dataset"
    outputs_root: str = "outputs"
    paths: PathsConfig = field(default_factory=PathsConfig)
    inference: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    beton: BetonConfig = field(default_factory=BetonConfig)


# ─────────── groups ───────────


@dataclass
class GroupsConfig:
    """Ensemble component discovery controls (used by steps 04/04b/04c/05)."""

    group_by: List[str] = field(default_factory=lambda: ["topology", "rho"])
    min_components: int = 2
    filter: Any = field(default_factory=dict)
    sample_size: Optional[int] = None
    field_ranges: Any = field(default_factory=dict)
    sampling_strategy: str = "combinatorial"  # combinatorial | monte_carlo
    mc_n_samples: Optional[int] = None
    mc_seed: Optional[int] = None


# ─────────── profiling ───────────


@dataclass
class ProfilingAnchorsConfig:
    per_class: int = 100
    strategy: str = "per_class_first_n"
    order_by: str = "example_id"
    source_split: str = "test"


@dataclass
class ProfilingProfilesConfig:
    """Category similarity profile computation (step 03)."""

    skip: bool = False
    metrics: List[str] = field(default_factory=lambda: ["cosine", "l2"])


@dataclass
class ProfilingDiagnosticsConfig:
    """Per-model diagnostic metrics (step 03b)."""

    morans_i: bool = True
    weight_norms: bool = True
    unit_distance_correlation: bool = True


@dataclass
class ProfilingConfig:
    anchors: ProfilingAnchorsConfig = field(default_factory=ProfilingAnchorsConfig)
    profiles: ProfilingProfilesConfig = field(default_factory=ProfilingProfilesConfig)
    diagnostics: ProfilingDiagnosticsConfig = field(
        default_factory=ProfilingDiagnosticsConfig
    )


# ─────────── analysis ───────────


@dataclass
class DiversityConfig:
    """Post-ensemble diversity metrics (step 04b)."""

    enabled: bool = True
    metrics: List[str] = field(
        default_factory=lambda: [
            "q_statistic",
            "disagreement",
            "double_fault",
            "correlation",
            "interrater_agreement",
        ]
    )


@dataclass
class ConsistencyConfig:
    """Post-ensemble RDM/RSA consistency (step 04c)."""

    enabled: bool = True


@dataclass
class AnalysisConfig:
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)


# ─────────── execution ───────────


@dataclass
class ExecutionConfig:
    """Common execution knobs shared across post-training pipeline steps."""

    split: str = "test"
    force: bool = False


# ─────────── mlflow ───────────


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///outputs/mlflow.db"
    artifact_location: str = "outputs/mlruns"
    artifact_cache_dir: str = "outputs/mlflow_artifact_cache"
    experiment_name: str = "contopo"
    enable_system_metrics: bool = True


# ─────────── ensemble ───────────


@dataclass
class EnsembleConfig:
    """Ensemble voting configuration. Discovery controls live in GroupsConfig."""

    name: str = "dynamic_ensembles"
    votes: List[str] = field(
        default_factory=lambda: ["soft", "hard", "max_confidence", "conf_weighted"]
    )


# ─────────── adapter ───────────


@dataclass
class MetaSplitFractionsConfig:
    train: float = 0.6
    val: float = 0.2
    holdout: float = 0.2


@dataclass
class MetaSplitConfig:
    seed: int = 42
    fractions: MetaSplitFractionsConfig = field(
        default_factory=MetaSplitFractionsConfig
    )


@dataclass
class AnchorSelectionConfig:
    """Anchor selection used by meta-learners."""

    per_class: int = 100
    strategy: str = "per_class_first_n"
    order_by: str = "example_id"


@dataclass
class AdapterConfig:
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 256
    bias: bool = True
    dropout: float = 0.3
    meta_type: str = "meta_lr"
    feature_type: str = "logits"
    similarity_metric: str = "cosine"
    profile_mask: str = "true_class"
    hidden_dim: int = 128
    init_seed: int = 42
    meta_split: MetaSplitConfig = field(default_factory=MetaSplitConfig)
    anchor_selection: AnchorSelectionConfig = field(
        default_factory=AnchorSelectionConfig
    )


# ─────────── pipeline ───────────


@dataclass
class PipelineStepConfig:
    id: str = MISSING
    script: str = MISSING
    sweep: Optional[str] = None
    overrides: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class PipelineConfig:
    steps: List[Any] = field(default_factory=list)
    from_step: Optional[str] = None


# ─────────── top-level ───────────


@dataclass
class ConTopoConfig:
    schema_version: int = 1
    trial: int = 0
    seed: Optional[int] = None

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    groups: GroupsConfig = field(default_factory=GroupsConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# ─────────── ConfigStore registration ───────────


def register_configs() -> None:
    """Register all structured configs in Hydra's ConfigStore."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    # Top-level schema
    cs.store(name="base_config", node=ConTopoConfig)

    # Group schemas
    cs.store(group="model", name="base_resnet18", node=ModelConfig)
    cs.store(group="loss", name="base_cross_entropy", node=LossConfig)
    cs.store(group="dataset", name="base_cifar10", node=DatasetConfig)
    cs.store(group="training", name="base_default", node=TrainingConfig)
    cs.store(group="runtime", name="base_default", node=RuntimeConfig)
    cs.store(group="groups", name="base_default", node=GroupsConfig)
    cs.store(group="profiling", name="base_default", node=ProfilingConfig)
    cs.store(group="analysis", name="base_default", node=AnalysisConfig)
    cs.store(group="execution", name="base_default", node=ExecutionConfig)
    cs.store(group="mlflow", name="base_default", node=MLflowConfig)
    cs.store(group="ensemble", name="base_dynamic", node=EnsembleConfig)
    cs.store(group="adapter", name="base_default", node=AdapterConfig)
    cs.store(group="pipeline", name="base_default", node=PipelineConfig)
