# Pending implementations

Two training features required for the ResNet34/ImageNet100 paper replication
are not yet implemented in `scripts/01_train_models.py`:

## 1. ReduceLROnPlateau scheduler (`scheduler: plateau`)

`_build_scheduler` currently supports `cyclic`, `cosine`, and `none` only.
Needs a `"plateau"` branch wired to
`torch.optim.lr_scheduler.ReduceLROnPlateau`, stepped per-epoch on `val_loss`,
with `factor`, `patience`, and `min_lr` sourced from config. Requires new
fields in `TrainingConfig` and the YAML configs.

Paper values: factor=0.5, patience=4, min_lr=1e-6.

## 2. `min_delta` for early stopping

The improvement check is a bare `val_loss < best_val_loss` with no threshold.
The paper uses `min_delta=1e-4` (improvement only counts if the drop exceeds
this value). Needs a new `early_stopping_min_delta` field in `TrainingConfig`
(default `0.0` to keep existing runs unaffected).

Paper value: min_delta=1e-4.
