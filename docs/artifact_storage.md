# Artifact Storage Reference

Documents every artifact logged to MLflow by run kind, what it contains,
and a size estimate per run. Estimates assume float32 storage and standard
`np.savez_compressed` / PyTorch `.pt` compression ratios (~70–80% of raw).

Reference datasets used for estimates:

| Dataset | Test split | Classes | `embedding_dim` | Anchors (default profiling) |
|---|---|---|---|---|
| CIFAR-10 | 10 000 samples | 10 | 256 | 100 × 10 = 1 000 |
| ImageNet100 | 5 000 samples | 100 | 256 | 100 × 100 = 10 000 |

---

## `model` run — `scripts/01_train_models.py`

| Artifact | Content | CIFAR-10 | ImageNet100 |
|---|---|---|---|
| `e2e_best/` | Full PyTorch model directory (weights + MLflow metadata) | ~45 MB | ~85 MB |
| `config/resolved_config.yaml` | Fully resolved Hydra config | ~20 KB | ~20 KB |

**Per-run total: ~45 MB (CIFAR-10) / ~85 MB (ImageNet100)**

Notes:
- `e2e_best/` size scales with parameter count: ResNet18 (~11 M params) ≈ 45 MB, ResNet34 (~21 M params) ≈ 85 MB.
- Periodic checkpoints (`checkpoint_epoch*/`) are **off by default** (`save_checkpoints: false`). When enabled, each checkpoint is the same size as `e2e_best/`.
- `config/resolved_config.yaml` is read by migration scripts and must be kept.

---

## `inference` run — `scripts/02_cache_inference.py`

| Artifact | Content | CIFAR-10 (test, 10 k) | ImageNet100 (test, 5 k) |
|---|---|---|---|
| `inference/{split}_inference_results.parquet` | `original_index`, `label`, `prediction` (3 columns) | ~100 KB | ~50 KB |
| `inference/{split}_tensors.npz` | `embeddings` [N × 256], `logits` [N × C] | ~8–9 MB | ~5 MB |

**Per-run total: ~9 MB (CIFAR-10) / ~5 MB (ImageNet100)**

Notes:
- One run per (model × split). In typical usage only the `test` split is cached, so one run per model.
- `embeddings` dominates (N × 256 × 4 B); `logits` is small (N × C × 4 B).
- `probs` was removed in the S3 cleanup — it is derivable as `softmax(logits)` on demand.

---

## `category_similarity_profile` run — `scripts/03_compute_profiles.py`

| Artifact | Content | CIFAR-10 | ImageNet100 |
|---|---|---|---|
| `profiles/{split}_{metric}_profiles.pt` | Similarity matrix [N\_samples × N\_anchors] | ~30–35 MB | ~150–200 MB |

**Per-run total: ~30–35 MB (CIFAR-10) / ~150–200 MB (ImageNet100)**

Notes:
- One run per (model × split × similarity metric). Default: one metric (`cosine`), one split (`test`).
- The profile matrix is **N\_samples × N\_anchors** where N\_anchors = `anchors.per_class × num_classes` (default 1 000 for CIFAR-10, 10 000 for ImageNet100). ImageNet100 profiles are large because of the 10× more classes.
- The `config/resolved_config.yaml` artifact is also logged here (~20 KB).

---

## `diagnostics` run — `scripts/03b_compute_diagnostics.py`

| Artifact | Content | Size |
|---|---|---|
| `diagnostics/weight_norms.pt` *(optional)* | Per-unit L2 weight norms [256] | < 5 KB |
| `diagnostics/unit_distance_correlation.pt` *(optional)* | Grid-distance vs weight-similarity correlation [256] | < 5 KB |

**Per-run total: < 10 KB**

Notes:
- Both artifacts are optional and only written when the respective diagnostic metric is enabled.
- One run per (model × diagnostic\_metric). Purely analysis-facing; no pipeline stage reads these.

---

## `ensemble` run — `scripts/04_run_ensemble.py`

| Artifact | Content | CIFAR-10 (test, 10 k) | ImageNet100 (test, 5 k) |
|---|---|---|---|
| `ensemble/{split}_{name}_{method}_inference.parquet` | `original_index`, `label`, `prediction` | ~100 KB | ~50 KB |
| `ensemble/{split}_{name}_{method}_tensors.npz` *(optional)* | Ensemble `probs` [N × C] | ~200 KB | ~1 MB |

**Per-run total: ~300 KB (CIFAR-10) / ~1 MB (ImageNet100)**

Notes:
- One run per (ensemble\_name × voting\_method). Typical pipeline: 2–3 methods per ensemble.
- `composition_map.json` was removed in the S3 cleanup. Component mapping is recoverable from `component_run_ids_csv` tag + each inference run's `trained_model_run_id` tag.
- The `tensors.npz` is optional in the schema; it stores the aggregated probability distribution for calibration and uncertainty analysis.

---

## `diversity` run — `scripts/04b_compute_diversity.py`

No artifacts logged. Diversity metrics are stored as MLflow metrics only.

**Per-run total: 0 B**

---

## `consistency` run — `scripts/04c_compute_consistency.py`

| Artifact | Content | Size |
|---|---|---|
| `consistency/rsa_matrix.pt` | RSA correlation matrix [N\_components × N\_components] | < 5 KB (for ≤ 20 components) |
| `consistency/run_id_ordering.json` | Ordered list of run IDs (matrix row/col key) | < 2 KB |

**Per-run total: < 10 KB**

---

## `metalearner` run — `scripts/05_train_adapters.py`

| Artifact | Content | CIFAR-10 (logits) | CIFAR-10 (embeddings) | ImageNet100 (logits) |
|---|---|---|---|---|
| `inputs/adapter_inputs_{hash}.npz` | X\_train/val/holdout, y arrays, split indices, standardization params | ~5 MB | ~100–150 MB | ~3 MB |
| `data/adapter_holdout_{hash}.parquet` | `original_index`, `label`, `prediction`, `confidence` | ~50 KB | ~50 KB | ~25 KB |
| `data/adapter_holdout_{hash}.npz` | Holdout `probs` [N\_holdout × C] | ~50 KB | ~50 KB | ~200 KB |
| `model/` | Trained adapter (small MLP or linear head) | ~1–5 MB | ~1–5 MB | ~1–5 MB |
| `config/resolved_config.yaml` | Fully resolved Hydra config | ~20 KB | ~20 KB | ~20 KB |

**Per-run total: ~6–10 MB (logits, CIFAR-10) / ~105–160 MB (embeddings, CIFAR-10)**

Notes:
- Feature dimensionality drives `adapter_inputs_{hash}.npz` size: `feature_type=logits` gives N\_components × C features (e.g. 9 × 10 = 90), while `feature_type=embeddings` gives N\_components × 256 (e.g. 9 × 256 = 2 304).
- `adapter_split_trace_{hash}.parquet` was removed in the S3 cleanup. Split assignments are already stored as `train_idx`/`val_idx`/`holdout_idx` arrays inside `adapter_inputs_{hash}.npz`.
- `data/adapter_holdout_{hash}.npz` stores the full probability distribution on the holdout set for future calibration analysis.

---

## Summary by run kind

| Kind | CIFAR-10 per run | ImageNet100 per run | Multiplier |
|---|---|---|---|
| `model` | ~45 MB | ~85 MB | 1× per (architecture × seed × rho × topology) |
| `inference` | ~9 MB | ~5 MB | 1× per (model × split) |
| `category_similarity_profile` | ~33 MB | ~175 MB | 1× per (model × split × metric) |
| `diagnostics` | < 10 KB | < 10 KB | 1× per (model × diagnostic\_metric) |
| `ensemble` | ~300 KB | ~1 MB | 1× per (ensemble\_name × voting\_method) |
| `diversity` | 0 | 0 | 1× per (ensemble × diversity\_metric) |
| `consistency` | < 10 KB | < 10 KB | 1× per ensemble |
| `metalearner` (logits) | ~8 MB | ~5 MB | 1× per (ensemble × meta\_type) |
| `metalearner` (embeddings) | ~130 MB | ~80 MB | 1× per (ensemble × meta\_type) |

---

## What was removed in the S3 cleanup

These artifacts were present in earlier runs but are no longer written:

| Artifact | Kind | Why removed |
|---|---|---|
| `probs` field in `inference/{split}_tensors.npz` | `inference` | Never read downstream; derivable as `softmax(logits)` |
| `ensemble/composition_map.json` | `ensemble` | Never read; component mapping reconstructible from tags |
| `inputs/adapter_split_trace_{hash}.parquet` | `metalearner` | Fully redundant with `train_idx`/`val_idx`/`holdout_idx` in `adapter_inputs_{hash}.npz` |

Old runs on S3 still have these artifacts. To permanently free that storage,
run `mlflow gc` after soft-deleting the relevant runs — see
`scripts/migrations/delete_non_model_runs.py` for the deletion script and gc
instructions.
