import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch

from utils.train import unwrap
from networks.shallowCNN import LinearShallowCNN, ProjectionShallowCNN, LinearClassifier
from networks.modified_ResNet18 import LinearResNet18, ProjectionResNet18


@dataclass
class LoadedModelBundle:
    encoder: torch.nn.Module
    classifier: Optional[torch.nn.Module]
    meta: dict


_E2E_CKPT_ORDER = {
    "best": ("e2e_best.pth", "e2e_last.pth"),
    "last": ("e2e_last.pth", "e2e_best.pth"),
}
_CONTRASTIVE_ENCODER_ORDER = ("contrastive_last.pth", "contrastive_best.pth")
_READOUT_ORDER = ("readout_best.pth", "readout_last.pth")


def _normalize_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _first_existing(run_folder: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = os.path.join(run_folder, name)
        if os.path.isfile(path):
            return path
    return None


def _prepare_module(
    module: torch.nn.Module | None,
    device: torch.device,
    dp_if_multi_gpu: bool,
    eval_mode: bool,
) -> torch.nn.Module | None:
    if module is None:
        return None
    module = module.to(device)
    if dp_if_multi_gpu and torch.cuda.device_count() > 1:
        module = torch.nn.DataParallel(module)
    if eval_mode:
        module.eval()
    return module

def _maybe_fix_state_dict_keys(state_dict: dict, model: torch.nn.Module) -> dict:
    """Normalize DataParallel prefixes: add/remove 'module.' to match `model`."""
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    is_dp = isinstance(model, torch.nn.DataParallel)
    if has_module and not is_dp:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if (not has_module) and is_dp:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _infer_stage(ckpt: dict, ckpt_path: str) -> str:
    """
    Decide whether the checkpoint is 'e2e' (CE) or 'contrastive'.

    Priority:
    1) Use ckpt['stage'] if present.
    2) Else infer from filename.
    3) Else infer from args (presence of 'projection_dim' → contrastive).
    """
    stage = ckpt.get("stage")
    if stage in {"e2e", "contrastive"}:
        return stage
    name = os.path.basename(ckpt_path).lower()
    if "contrastive" in name:
        return "contrastive"
    args = ckpt.get("args", {})

    return "contrastive" if "projection_dim" in args else "e2e"


def _build_head_from_args(args: dict, stage: str, device: torch.device, dp_if_multi_gpu: bool):
    """
    Recreate the SAME head used at train time (Linear* for CE, Projection* for contrastive),
    so weights load cleanly and we can then extract `.encoder`.
    """
    model_type = args.get("model_type", "shallowcnn")
    emb_dim = int(args.get("embedding_dim", 256))

    if stage == "contrastive":
        proj_dim = int(args.get("projection_dim", 128))
        if model_type == "resnet18":
            model = ProjectionResNet18(emb_dim=emb_dim, feat_dim=proj_dim, ret_emb=True)
        else:
            model = ProjectionShallowCNN(emb_dim=emb_dim, feat_dim=proj_dim, ret_emb=True, use_dropout=False)
    else:  # 'e2e' CE
        num_classes = int(args.get("num_classes", 10))
        if model_type == "resnet18":
            model = LinearResNet18(emb_dim=emb_dim, num_classes=num_classes, ret_emb=True)
        else:
            model = LinearShallowCNN(emb_dim=emb_dim, num_classes=num_classes, ret_emb=True, use_dropout=False)

    if torch.cuda.is_available():
        model = model.to(device)
        if dp_if_multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    return model


def _load_e2e_bundle(
    run_folder: str,
    device: torch.device,
    prefer: str,
    dp_if_multi_gpu: bool,
    eval_mode: bool,
    strict: bool,
) -> LoadedModelBundle | None:
    order = _E2E_CKPT_ORDER.get(prefer, _E2E_CKPT_ORDER["best"])
    ckpt_path = _first_existing(run_folder, order)
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = _build_head_from_args(args, "e2e", device, dp_if_multi_gpu)

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint missing 'state_dict': {ckpt_path}")

    state_dict = _maybe_fix_state_dict_keys(state_dict, model)
    model.load_state_dict(state_dict, strict=strict)

    base = unwrap(model)
    encoder_module = base.encoder
    classifier_module = getattr(base, "fc", None)

    encoder = _prepare_module(encoder_module, device, dp_if_multi_gpu, eval_mode)
    classifier = _prepare_module(classifier_module, device, dp_if_multi_gpu, eval_mode)

    meta = {
        "stage": "e2e",
        "run_folder": run_folder,
        "encoder_ckpt": ckpt_path,
        "classifier_ckpt": ckpt_path,
        "encoder_epoch": ckpt.get("epoch"),
        "classifier_epoch": ckpt.get("epoch"),
        "args": args,
        "metrics": ckpt.get("metrics"),
        "ckpt_path": ckpt_path,
        "figure_source": ckpt_path,
    }
    return LoadedModelBundle(encoder=encoder, classifier=classifier, meta=meta)


def _load_contrastive_bundle(
    run_folder: str,
    device: torch.device,
    prefer: str,
    dp_if_multi_gpu: bool,
    eval_mode: bool,
    strict: bool,
) -> LoadedModelBundle | None:
    ckpt_path = _first_existing(run_folder, _CONTRASTIVE_ENCODER_ORDER)
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = _build_head_from_args(args, "contrastive", device, dp_if_multi_gpu)

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint missing 'state_dict': {ckpt_path}")

    state_dict = _maybe_fix_state_dict_keys(state_dict, model)
    model.load_state_dict(state_dict, strict=strict)

    base = unwrap(model)
    encoder_module = base.encoder
    encoder = _prepare_module(encoder_module, device, dp_if_multi_gpu, eval_mode)

    if prefer == "last":
        readout_order = ("readout_last.pth", "readout_best.pth")
    else:
        readout_order = _READOUT_ORDER
    readout_path = _first_existing(run_folder, readout_order)
    if readout_path is None:
        raise FileNotFoundError(
            f"No linear readout checkpoint found in {run_folder}. "
            "Expected one of: readout_best.pth, readout_last.pth"
        )

    readout_ckpt = torch.load(readout_path, map_location=device, weights_only=False)
    linear_state = readout_ckpt.get("linear_state_dict")
    if linear_state is None:
        raise KeyError(f"Checkpoint missing 'linear_state_dict': {readout_path}")

    readout_args = readout_ckpt.get("args", {})
    emb_dim = int(readout_args.get("embedding_dim", args.get("embedding_dim", 256)))
    num_classes = int(readout_args.get("num_classes", args.get("num_classes", 10)))

    classifier_module = LinearClassifier(emb_dim=emb_dim, num_classes=num_classes)
    classifier_module = _prepare_module(classifier_module, device, dp_if_multi_gpu, eval_mode)

    linear_state = _maybe_fix_state_dict_keys(linear_state, classifier_module)
    classifier_module.load_state_dict(linear_state, strict=strict)

    meta = {
        "stage": "contrastive",
        "run_folder": run_folder,
        "encoder_ckpt": ckpt_path,
        "classifier_ckpt": readout_path,
        "encoder_epoch": ckpt.get("epoch"),
        "classifier_epoch": readout_ckpt.get("epoch"),
        "args": args,
        "readout_args": readout_args,
        "metrics": ckpt.get("metrics"),
        "readout_metrics": {
            "val_acc": readout_ckpt.get("val_acc"),
            "val_loss": readout_ckpt.get("val_loss"),
        },
        "ckpt_path": ckpt_path,
        "figure_source": ckpt_path,
    }
    return LoadedModelBundle(encoder=encoder, classifier=classifier_module, meta=meta)


def load_model_bundles(
    path: str,
    prefer: str = "best",
    device: str | torch.device | None = None,
    dp_if_multi_gpu: bool = False,
    eval_mode: bool = True,
    strict: bool = True,
) -> list[LoadedModelBundle]:
    if prefer not in _E2E_CKPT_ORDER:
        raise ValueError("'prefer' must be 'best' or 'last'")

    device_t = _normalize_device(device)
    path_abs = os.path.abspath(path)

    if os.path.isdir(path_abs):
        if _dir_contains_expected_ckpts(path_abs):
            run_folders = [path_abs]
        else:
            run_folders = list_run_folders_from_model_folder(path_abs)
    elif os.path.isfile(path_abs):
        run_folders = [os.path.dirname(path_abs)]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    bundles: list[LoadedModelBundle] = []
    for run in run_folders:
        bundle = _load_e2e_bundle(run, device_t, prefer, dp_if_multi_gpu, eval_mode, strict)
        if bundle is None:
            bundle = _load_contrastive_bundle(run, device_t, prefer, dp_if_multi_gpu, eval_mode, strict)
        if bundle is None:
            raise FileNotFoundError(
                f"No supported checkpoints found in {run}. "
                "Expected e2e_best/e2e_last or contrastive_*/readout_* pairs."
            )
        bundle.meta.setdefault("prefer", prefer)
        bundles.append(bundle)

    return bundles

def load_encoder_from_ckpt(
    ckpt_path: str,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
    dp_if_multi_gpu: bool = False,
    strict: bool = True,
):
    """
    Load ONLY the encoder (ShallowCNN/ResNet18) from a single checkpoint.

    Supported sources
    -----------------
    • CE (end-to-end) checkpoints: e2e_*.pth  → Linear* wrapper
    • Contrastive checkpoints:     contrastive_*.pth → Projection* wrapper

    Returns
    -------
    encoder : torch.nn.Module
        The backbone encoder moved to `device`. Wrapped in DataParallel if requested and available.
    meta : dict
        Minimal metadata: {'epoch', 'stage', 'args', 'metrics', 'ckpt_path'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stage = _infer_stage(ckpt, ckpt_path)

    # Reject linear-readout-only checkpoints (no encoder weights inside)
    if stage == "linear_readout" or ("linear_state_dict" in ckpt and "state_dict" not in ckpt):
        raise ValueError(
            "This checkpoint contains only the linear readout head (no encoder). "
            "Load a 'contrastive_*' or 'e2e_*' checkpoint instead."
        )

    args = ckpt.get("args", {})
    head = _build_head_from_args(args, stage, device, dp_if_multi_gpu)

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'state_dict' with encoder+head weights.")

    # Align DP prefixes and load weights
    state_dict = _maybe_fix_state_dict_keys(state_dict, head)
    head.load_state_dict(state_dict, strict=strict)

    # Extract ONLY the encoder (unwrap in case of DataParallel)
    encoder = unwrap(head).encoder

    if torch.cuda.is_available():
        encoder = encoder.to(device)
        if dp_if_multi_gpu and torch.cuda.device_count() > 1:
            encoder = torch.nn.DataParallel(encoder)

    if eval_mode:
        encoder.eval()

    meta = {
        "epoch": ckpt.get("epoch"),
        "stage": stage,
        "args": args,
        "metrics": ckpt.get("metrics"),
        "ckpt_path": ckpt_path,
    }
    return encoder, meta


def load_encoder_from_run_folder(
    run_folder: str,
    prefer: str = "best",
    device: str | torch.device | None = None,
    **kwargs,
):
    """
    Convenience helper: pick a good checkpoint from a run folder and load the encoder.

    Selection order
    ---------------
    prefer='best': contrastive_best.pth → e2e_best.pth → contrastive_last.pth → e2e_last.pth
    prefer='last': contrastive_last.pth → e2e_last.pth → contrastive_best.pth → e2e_best.pth
    """
    order_best = ["contrastive_best.pth", "e2e_best.pth", "contrastive_last.pth", "e2e_last.pth"]
    order_last = ["contrastive_last.pth", "e2e_last.pth", "contrastive_best.pth", "e2e_best.pth"]
    candidates = order_best if prefer == "best" else order_last

    for fname in candidates:
        ckpt_path = os.path.join(run_folder, fname)
        if os.path.isfile(ckpt_path):
            return load_encoder_from_ckpt(ckpt_path, device=device, **kwargs)

    raise FileNotFoundError(
        f"No suitable checkpoint found in {run_folder}. "
        "Expected one of: contrastive_best/last.pth or e2e_best/last.pth"
    )

def load_encoder_from_path(path: str, device: str, prefer: str, dp: bool):
    """Accept either a checkpoint file or a run folder and return (encoder, meta)."""
    if os.path.isdir(path):
        encoder, meta = load_encoder_from_run_folder(
            run_folder=path,
            prefer=prefer,
            device=device,
            dp_if_multi_gpu=dp,
            eval_mode=True,
        )
    else:
        encoder, meta = load_encoder_from_ckpt(
            ckpt_path=path,
            device=device,
            dp_if_multi_gpu=dp,
            eval_mode=True,
        )
    return encoder, meta

EXPECTED_CKPT_NAMES = {
    "contrastive_best.pth",
    "contrastive_last.pth",
    "e2e_best.pth",
    "e2e_last.pth",
}


def _dir_contains_expected_ckpts(d: str) -> bool:
    """Return True only if directory has one of the expected run checkpoint files."""
    if not os.path.isdir(d):
        return False
    try:
        entries = os.listdir(d)
    except FileNotFoundError:
        return False
    return any(name in EXPECTED_CKPT_NAMES for name in entries)

def list_run_folders_from_model_folder(model_folder: str) -> list[str]:
    """
    Given a model folder that may aggregate multiple trials, return a list of
    run folders (each containing checkpoints).

    Handles both layouts:
    - Nested trials: <model_folder>/trial_00, trial_01, ... (each with *.pth)
    - Flat trial folder: <model_folder> itself contains *.pth
    """
    model_folder = os.path.abspath(model_folder)
    # If the model folder itself contains expected checkpoint files, treat it as a run folder.
    if _dir_contains_expected_ckpts(model_folder):
        return [model_folder]
    # Otherwise, look for immediate child dirs that contain checkpoints
    runs: list[str] = []
    for name in sorted(os.listdir(model_folder)):
        d = os.path.join(model_folder, name)
        if os.path.isdir(d) and _dir_contains_expected_ckpts(d):
            runs.append(d)
    if runs:
        return runs
    raise FileNotFoundError(
        f"No checkpoints found under model folder: {model_folder}. "
        "Expected *.pth files directly or inside subfolders (e.g., trial_00)."
    )

def load_encoders_from_model_folder(
    model_folder: str,
    prefer: str = "best",
    device: str | torch.device | None = None,
    dp_if_multi_gpu: bool = False,
    eval_mode: bool = True,
    strict: bool = True,
):
    """
    Load encoders for all trials inside a given model folder.

    Returns a list of (encoder, meta) for each discovered run folder.
    """
    bundles = load_model_bundles(
        path=model_folder,
        prefer=prefer,
        device=device,
        dp_if_multi_gpu=dp_if_multi_gpu,
        eval_mode=eval_mode,
        strict=strict,
    )
    return [(bundle.encoder, bundle.meta) for bundle in bundles]

def parse_model_load_args():
    parser = argparse.ArgumentParser(
        description="Load a trained encoder (from CE or contrastive) and run eval-time experiments."
    )
    parser.add_argument(
        "path",
        help=(
            "Relative path to a checkpoint file (e2e_*.pth / contrastive_*.pth) "
            "or a run folder that contains them."
        ),
    )
    parser.add_argument(
        "--prefer",
        choices=["best", "last"],
        default="best",
        help=(
            "When 'path' resolves to a run folder, choose which checkpoint combination to load. "
            "Default: 'best' selects validation-best CE checkpoints or the contrastive_last/readout_best pair for contrastive runs."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="Wrap the returned encoder in DataParallel if multiple GPUs are available.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dataset-root", default="./dataset")
    return parser.parse_args()
