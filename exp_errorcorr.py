"""Evaluate pairwise error correlations across model trials.

This script runs a classification experiment for every model folder contained in
the path supplied on the command line.  It leverages the unified loading helper
`load_model_bundles` so both cross-entropy and contrastive+readout checkpoints
are supported transparently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from utils.load import parse_model_load_args, load_model_bundles
from utils.experiments import CIFAR10_MEAN, CIFAR10_STD


def _collect_errors_and_preds(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Allocate per-sample buffers up front so batches can write into them without concatenation.
    total = len(loader.dataset)
    errors = torch.zeros(total, dtype=torch.float32)
    preds = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    offset = 0
    logits_store: torch.Tensor | None = None

    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        for batch in loader:
            # Some collates add extra metadata; slice to the canonical (images, labels).
            images, labels = batch[:2] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = encoder(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            logits = classifier(feats)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]

            batch_preds = logits.argmax(dim=1)
            batch_errors = (batch_preds != labels).float().cpu()

            size = batch_errors.numel()
            errors[offset : offset + size] = batch_errors
            preds[offset : offset + size] = batch_preds.cpu()
            targets[offset : offset + size] = labels.cpu()
            logits_cpu = logits.detach().cpu()
            if logits_store is None:
                # Lazily size the logits tensor once we know how many classes the head emits.
                logits_store = torch.empty(total, logits_cpu.size(1), dtype=logits_cpu.dtype)
            logits_store[offset : offset + size] = logits_cpu
            offset += size

    if logits_store is None:
        logits_store = torch.empty(total, 0)

    return errors, preds, targets, logits_store


def _pearson_corrcoef(error_matrix: torch.Tensor) -> torch.Tensor:
    # Rows correspond to trials; compute their pairwise Pearson correlations over samples.
    if error_matrix.size(0) == 0:
        return torch.empty(0, 0)
    centered = error_matrix - error_matrix.mean(dim=1, keepdim=True)
    cov = centered @ centered.T
    var = centered.pow(2).sum(dim=1)
    denom = torch.sqrt(var).unsqueeze(0) * torch.sqrt(var).unsqueeze(1)
    corr = cov.clone()
    mask = denom == 0
    corr[~mask] = corr[~mask] / denom[~mask]
    corr.masked_fill_(mask, float("nan"))
    idx = torch.arange(corr.size(0))
    corr[idx, idx] = 1.0
    return corr


def ensemble_accuracy(
    logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    method: str = "soft",
) -> float:
    """Replicate professor's ensemble routines using numpy for parity."""

    if not logits_list:
        raise ValueError("logits_list must contain at least one model output")

    logits_np = [logits.detach().cpu().numpy() for logits in logits_list]
    labels_np = labels.detach().cpu().numpy()

    num_samples, num_classes = logits_np[0].shape
    for logits in logits_np:
        if logits.shape != (num_samples, num_classes):
            raise ValueError("All logits must share the same shape")

    probs = [np.exp(l) / np.exp(l).sum(axis=1, keepdims=True) for l in logits_np]

    if method == "hard":
        preds = np.array([np.argmax(l, axis=1) for l in logits_np])
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=num_classes).argmax(), axis=0, arr=preds
        )
    elif method == "soft":
        avg_probs = np.mean(probs, axis=0)
        final_preds = np.argmax(avg_probs, axis=1)
    elif method == "max_confidence":
        probs_stack = np.stack(probs, axis=0)
        max_conf = probs_stack.max(axis=2)
        best_model = np.argmax(max_conf, axis=0)
        final_preds = np.array(
            [np.argmax(probs_stack[best_model[i], i]) for i in range(num_samples)]
        )
    elif method == "conf_weighted":
        probs_stack = np.stack(probs, axis=0)
        confs = probs_stack.max(axis=2)
        weights = confs / confs.sum(axis=0, keepdims=True)
        weighted_probs = np.einsum("mn,mnc->nc", weights, probs_stack)
        final_preds = np.argmax(weighted_probs, axis=1)
    else:
        raise ValueError(
            "Unknown method. Choose from ['hard', 'soft', 'max_confidence', 'conf_weighted']."
        )

    return float(np.mean(final_preds == labels_np))


_PINK_FREQ_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _pink_frequency_radius(height: int, width: int) -> np.ndarray:
    key = (height, width)
    cached = _PINK_FREQ_CACHE.get(key)
    if cached is not None:
        return cached

    f_x = np.fft.fftfreq(width)
    f_y = np.fft.fftfreq(height)
    f_xx, f_yy = np.meshgrid(f_x, f_y)
    radius = np.sqrt(f_xx**2 + f_yy**2)
    radius[0, 0] = 1.0
    _PINK_FREQ_CACHE[key] = radius
    return radius


def _apply_white_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    if noise_level <= 0:
        return images.clone()
    noisy = images + noise_level * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)


def _apply_pink_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    if noise_level <= 0:
        return images.clone()

    batch_size, channels, height, width = images.shape
    radius = _pink_frequency_radius(height, width)
    noisy = torch.empty_like(images)

    for idx in range(batch_size):
        # Build 1/f noise in the Fourier domain for every channel independently.
        channel_noises = []
        for _ in range(channels):
            white = np.random.normal(0.0, 1.0, (height, width))
            f_white = np.fft.fft2(white)
            pink = np.real(np.fft.ifft2(f_white / radius))
            pink -= pink.mean()
            std = pink.std()
            if std == 0:
                std = 1.0
            pink = pink / std
            channel_noises.append(torch.from_numpy(pink).to(dtype=images.dtype))
        noise = torch.stack(channel_noises, dim=0)
        noisy[idx] = noise

    noisy = images + noise_level * noisy
    return torch.clamp(noisy, 0.0, 1.0)


def _apply_salt_pepper_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    if noise_level <= 0:
        return images.clone()

    noisy = images.clone()
    batch_size, channels, height, width = noisy.shape
    num_pixels = height * width
    num_corrupt = int(noise_level * num_pixels)
    if num_corrupt <= 0:
        return noisy

    device = noisy.device
    for idx in range(batch_size):
        # Corrupt random pixels per image, toggling them to black or white with equal probability.
        coords = torch.randint(0, num_pixels, (num_corrupt,), device=device)
        rows = coords // width
        cols = coords % width
        salt_mask = torch.rand(num_corrupt, device=device) >= 0.5

        img = noisy[idx]
        if salt_mask.any():
            r = rows[salt_mask]
            c = cols[salt_mask]
            img[:, r, c] = 1.0
        pepper_mask = ~salt_mask
        if pepper_mask.any():
            r = rows[pepper_mask]
            c = cols[pepper_mask]
            img[:, r, c] = 0.0

    return noisy


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images - mean) / std


def _load_clean_test_images(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    images_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)

    # Materialize the entire test split so downstream noise sampling can operate in-memory.
    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return images, labels


def _build_noisy_loader(
    base_images: torch.Tensor,
    base_labels: torch.Tensor,
    noise_type: str | None,
    noise_level: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    if noise_type is None or noise_level <= 0:
        noisy = base_images.clone()
    elif noise_type == "white":
        noisy = _apply_white_noise(base_images, noise_level)
    elif noise_type == "pink":
        noisy = _apply_pink_noise(base_images, noise_level)
    elif noise_type == "salt_pepper":
        noisy = _apply_salt_pepper_noise(base_images, noise_level)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    normalized = _normalize_images(noisy)
    # Pair the perturbed images with the original labels for evaluation.
    dataset = torch.utils.data.TensorDataset(normalized, base_labels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def _evaluate_bundles(
    bundles,
    loader: torch.utils.data.DataLoader,
) -> dict | None:
    # Accumulate per-trial results so we can compute correlations and ensembles later.
    errors_all: list[torch.Tensor] = []
    preds_all: list[torch.Tensor] = []
    counts: list[int] = []
    accuracies: list[float] = []
    run_names: list[str] = []
    logits_all: list[torch.Tensor] = []
    labels_ref: torch.Tensor | None = None

    for bundle in bundles:
        encoder = bundle.encoder
        classifier = bundle.classifier
        if classifier is None:
            print(f"Skipping {_run_name(bundle.meta)}: classifier head not found.")
            continue

        device = next(encoder.parameters()).device
        errors, preds, labels, logits = _collect_errors_and_preds(encoder, classifier, loader, device)

        if labels_ref is None:
            labels_ref = labels
        elif not torch.equal(labels_ref, labels):
            raise RuntimeError("Mismatched label ordering across trials.")

        errors_all.append(errors)
        preds_all.append(preds)
        logits_all.append(logits)
        # Track absolute error counts for quick diagnostics.
        counts.append(int(errors.sum().item()))
        accuracies.append(float((preds == labels).float().mean().item()))
        run_names.append(_run_name(bundle.meta))

    if not errors_all:
        return None

    # Stack errors into a matrix with shape (num_trials, num_samples).
    error_matrix = torch.stack(errors_all)

    # Stack logits into a matrix with shape (num_trials, num_samples, num_classes)
    logits_matrix = torch.stack(logits_all) if logits_all else None

    non_ensemble_mean = float(np.mean(accuracies)) if accuracies else float("nan")
    if not accuracies:
        non_ensemble_std = float("nan")
    elif len(accuracies) > 1:
        non_ensemble_std = float(np.std(accuracies, ddof=1))
    else:
        non_ensemble_std = 0.0

    ensemble_results: dict[str, float] = {}
    if labels_ref is not None and logits_all:
        # Evaluate each ensemble voting scheme against the shared label order.
        for method in ("soft", "hard", "max_confidence", "conf_weighted"):
            ensemble_results[method] = ensemble_accuracy(logits_all, labels_ref, method=method)

    return {
        "counts": counts,
        "run_names": run_names,
        "accuracies": accuracies,
        "error_matrix": error_matrix,
        "ensemble_results": ensemble_results,
        "non_ensemble_mean": non_ensemble_mean,
        "non_ensemble_std": non_ensemble_std,
        "logits_matrix": logits_matrix,
        "labels_ref": labels_ref,
    }


def _print_diagnostics(result: dict, label: str) -> None:
    # Emit rich per-condition statistics before the summary table is produced.
    print(f"Diagnostics ({label}):")

    print("Error counts per trial:")
    for name, count in zip(result["run_names"], result["counts"]):
        print(f"{name}: {count}")

    print("Individual accuracies:")
    for name, acc in zip(result["run_names"], result["accuracies"]):
        print(f"{name}: {acc:.4f}")

    mean = result["non_ensemble_mean"]
    std = result["non_ensemble_std"]
    if np.isfinite(mean):
        print(f"Non-ensemble mean +/- std: {mean:.4f} +/- {std:.4f}")

    corr = _pearson_corrcoef(result["error_matrix"])
    print("Correlation matrix:")
    print(corr.tolist())

    if corr.numel() and corr.size(0) > 1:
        idx = torch.triu_indices(corr.size(0), corr.size(1), offset=1)
        vals = corr[idx[0], idx[1]]
        vals = vals[torch.isfinite(vals)]
        if vals.numel():
            pair_mean = float(vals.mean().item())
            pair_std = float(vals.std(unbiased=vals.numel() > 1).item())
            print(f"Pairwise mean: {pair_mean}")
            print(f"Pairwise std: {pair_std}")
        else:
            print("Pairwise mean: nan")
            print("Pairwise std: nan")
    else:
        print("Pairwise mean: nan")
        print("Pairwise std: nan")

    ensembles = result["ensemble_results"]
    for method in ("soft", "hard", "max_confidence", "conf_weighted"):
        if method in ensembles:
            print(f"Ensemble accuracy ({method}): {ensembles[method]:.4f}")


def _run_name(meta: dict) -> str:
    run_folder = meta.get("run_folder")
    if run_folder:
        return Path(run_folder).name
    ckpt = meta.get("ckpt_path")
    if ckpt:
        return Path(ckpt).parent.name
    return "run"


def main() -> None:
    args = parse_model_load_args()

    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )

    pin_memory = torch.cuda.is_available()
    base_images, base_labels = _load_clean_test_images(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
    noise_types = ["white", "pink", "salt_pepper"]

    # Enumerate the clean run and every noise/level combination we intend to probe.
    conditions: list[tuple[str, str | None, float]] = [("clean 0.00", None, 0.0)]
    for level in noise_levels[1:]:
        for noise_type in noise_types:
            conditions.append((f"{noise_type} {level:.2f}", noise_type, level))

    summary_rows: list[dict] = []

    for idx, (label, noise_type, noise_level) in enumerate(conditions):
        loader = _build_noisy_loader(
            base_images,
            base_labels,
            noise_type=noise_type,
            noise_level=noise_level,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        result = _evaluate_bundles(bundles, loader)
        if result is None:
            if idx == 0:
                print("No trials evaluated.")
            continue

        if idx == 0:
            # Show detailed metrics for the clean condition before the aggregate table.
            _print_diagnostics(result, label)

        summary_rows.append(
            {
                "condition": label,
                "non_ensemble_mean": result["non_ensemble_mean"],
                "non_ensemble_std": result["non_ensemble_std"],
                "ensemble_results": result["ensemble_results"],
            }
        )

    if not summary_rows:
        return

    headers = [
        "Condition",
        "Non-ensemble",
        "Soft Vote",
        "Hard Vote",
        "Max Conf",
        "Conf Weighted",
    ]

    table_rows: list[list[str]] = []
    for row in summary_rows:
        mean = row["non_ensemble_mean"]
        std = row["non_ensemble_std"]
        if np.isfinite(mean):
            non_ensemble_str = f"{mean:.4f} +/- {std:.4f}"
        else:
            non_ensemble_str = "nan"

        ensembles = row["ensemble_results"]
        soft = ensembles.get("soft")
        hard = ensembles.get("hard")
        max_conf = ensembles.get("max_confidence")
        conf_weighted = ensembles.get("conf_weighted")

        def _fmt(value: float | None) -> str:
            if value is None or not np.isfinite(value):
                return "nan"
            return f"{value:.4f}"

        table_rows.append(
            [
                row["condition"],
                non_ensemble_str,
                _fmt(soft),
                _fmt(hard),
                _fmt(max_conf),
                _fmt(conf_weighted),
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        # Expand column widths to fit the widest string seen per column.
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)

    print("\nAccuracy summary across conditions:")
    print(header_line)
    print(sep_line)
    for row in table_rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main()
