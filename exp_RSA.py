import os
import re
import argparse
import csv
from typing import List, Tuple, Dict, Any

import torch
import matplotlib.pyplot as plt


def _find_model_folders(models_root: str) -> List[str]:
    models_root = os.path.abspath(models_root)
    if not os.path.isdir(models_root):
        raise NotADirectoryError(f"Models root is not a directory: {models_root}")
    out: List[str] = []
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name)
        if os.path.isdir(path):
            out.append(path)
    return out


def _load_unaveraged_rdms(model_folder: str) -> torch.Tensor:
    """Load per-trial upper-triangular RDM vectors as [n_trials, m] float32 CPU."""
    base = os.path.basename(os.path.normpath(model_folder))
    rdm_file = os.path.join(model_folder, f"RDM_{base}.pt")
    if not os.path.isfile(rdm_file):
        raise FileNotFoundError(f"Missing unaveraged RDM file: {rdm_file}")

    obj = torch.load(rdm_file, map_location="cpu")

    if isinstance(obj, dict) and "rdms_upper" in obj:
        rdms = obj["rdms_upper"]
        if len(rdms) == 0:
            return torch.empty((0, 0), dtype=torch.float32)
        rdms_proc = [r.to(dtype=torch.float32, device="cpu").view(-1) for r in rdms]
        m = rdms_proc[0].numel()
        for i, r in enumerate(rdms_proc):
            if r.numel() != m:
                raise ValueError(
                    f"RDM vector length mismatch in {rdm_file}: index {i} has {r.numel()}, expected {m}"
                )
        return torch.stack(rdms_proc, dim=0)

    if torch.is_tensor(obj):
        T = obj.detach().to(dtype=torch.float32, device="cpu")
        if T.ndim == 1:
            T = T.view(1, -1)
        elif T.ndim != 2:
            raise ValueError(f"Unsupported RDM tensor shape in {rdm_file}: {tuple(T.shape)}")
        return T

    if isinstance(obj, (list, tuple)):
        rdms_proc = [torch.as_tensor(x, dtype=torch.float32, device="cpu").view(-1) for x in obj]
        if not rdms_proc:
            return torch.empty((0, 0), dtype=torch.float32)
        m = rdms_proc[0].numel()
        for i, r in enumerate(rdms_proc):
            if r.numel() != m:
                raise ValueError(
                    f"RDM vector length mismatch in {rdm_file}: index {i} has {r.numel()}, expected {m}"
                )
        return torch.stack(rdms_proc, dim=0)

    raise ValueError(f"Unrecognized RDM payload format in {rdm_file}")


def main():
    parser = argparse.ArgumentParser(description="Build model-by-model RSA over unaveraged RDMs.")
    parser.add_argument("models_root", nargs="?", default=".")
    parser.add_argument("--output-prefix", default="RSA")
    parser.add_argument("--trials-per-model", type=int, default=5,
                        help="Number of trials per model for collapsing (block-average). Default: 5")
    args = parser.parse_args()

    models_root = os.path.abspath(args.models_root)
    all_folders = _find_model_folders(models_root)

    # Only take models that contain '0.5dropout' (exclude 0.0 ones)
    # cand = [p for p in all_folders if "0.5dropout" in os.path.basename(p)]
    cand = all_folders

    # Sort by loss tag (prefix before first '_'), then numeric rho ascending, then name
    def _sort_key(path: str) -> Tuple[str, float, str]:
        name = os.path.basename(path.rstrip(os.sep))
        loss = name.split("_", 1)[0]
        m = re.search(r"_(\d+(?:\.\d+)?)rho(?:_|$)", name)
        rho = float(m.group(1)) if m else float("inf")
        return (loss, rho, name)

    model_folders = sorted(cand, key=_sort_key)
    if not model_folders:
        raise SystemExit(f"No matching model folders (containing '0.5dropout') under: {models_root}")

    # Aggregate rows and build index in the exact same order
    rows: List[torch.Tensor] = []
    index_rows: List[Dict[str, Any]] = []
    vec_len: int | None = None

    kept_models_meta: List[Dict[str, Any]] = []  # aligned with appended rows/models
    consistency_rows: List[Dict[str, Any]] = []
    for folder in model_folders:
        name = os.path.basename(folder.rstrip(os.sep))
        T = _load_unaveraged_rdms(folder)  # [n, m]
        if T.numel() == 0:
            continue
        n, m = T.shape
        if vec_len is None:
            vec_len = m
        elif m != vec_len:
            raise ValueError(f"RDM vector length mismatch for {name}: got {m}, expected {vec_len}")

        start = sum(t.shape[0] for t in rows)
        rows.append(T)
        # Capture meta for this model (once per model)
        loss = name.split("_", 1)[0]
        m_rho = re.search(r"_(\d+(?:\.\d+)?)rho(?:_|$)", name)
        rho_val = float(m_rho.group(1)) if m_rho else float("inf")
        kept_models_meta.append({"name": name, "loss": loss, "rho": rho_val})
        stats_path = os.path.join(folder, f"RDMConsistency_{name}.pt")
        if os.path.isfile(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
        else:
            stats = {}
        consistency_rows.append({
            "model": name,
            "mean": float(stats["mean"]) if "mean" in stats else "",
            "std": float(stats["std"]) if "std" in stats else "",
            "num_pairs": int(stats["num_pairs"]) if "num_pairs" in stats else "",
            "num_trials": int(stats["num_trials"]) if "num_trials" in stats else "",
            "message": stats.get("message", ""),
        })
        for i in range(n):
            index_rows.append({"global_index": start + i, "model": name, "trial_index": i})

    if not rows:
        raise SystemExit("No RDM rows aggregated. Nothing to save.")

    X = torch.cat(rows, dim=0).to(dtype=torch.float32, device="cpu")  # [g, m]
    g, m = X.shape

    # Pearson correlation across rows
    Xc = X - X.mean(dim=1, keepdim=True)
    eps = 1e-8
    norms = Xc.norm(dim=1, keepdim=True).clamp_min(eps)
    Y = Xc / norms
    rsa = (Y @ Y.t()).clamp(-1.0, 1.0)  # [g, g]

    # Save artifacts
    rsa_pt = os.path.join(models_root, f"{args.output_prefix}_{g}x{g}.pt")
    torch.save({"rsa_matrix": rsa, "index": index_rows}, rsa_pt)

    csv_path = os.path.join(models_root, f"{args.output_prefix}_index_{g}x{g}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_index", "model", "trial_index"]) 
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    consistency_csv = os.path.join(models_root, f"{args.output_prefix}_consistency.csv")
    with open(consistency_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mean", "std", "num_pairs", "num_trials", "message"])
        writer.writeheader()
        for row in consistency_rows:
            writer.writerow(row)

    png_path = os.path.join(models_root, f"{args.output_prefix}_{g}x{g}.png")
    plt.figure(figsize=(8, 7))
    im = plt.imshow(rsa.numpy(), cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
    plt.title(f"Model-by-model RSA ({g}x{g})")
    plt.xlabel("index")
    plt.ylabel("index")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Optionally collapse across trials by simple block-averaging (e.g., 5x5 blocks)
    t = int(args.trials_per_model)
    if t > 0 and g % t == 0:
        n_models = g // t
        # reshape to [models, trials, models, trials] and mean over trial dims
        rsa_collapsed = rsa.reshape(n_models, t, n_models, t).mean(dim=(1, 3))

        # Save collapsed matrix
        rsa_c_pt = os.path.join(models_root, f"{args.output_prefix}_collapsed_{n_models}x{n_models}.pt")
        torch.save({
            "rsa_matrix_collapsed": rsa_collapsed,
            "trials_per_model": t,
            "original_shape": (g, g),
            "collapsed_shape": (n_models, n_models),
        }, rsa_c_pt)

        # Plot collapsed heatmap
        png_c_path = os.path.join(models_root, f"{args.output_prefix}_collapsed_{n_models}x{n_models}.png")
        plt.figure(figsize=(7, 6))
        im = plt.imshow(rsa_collapsed.numpy(), cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
        plt.title(f"Model-by-model RSA collapsed ({n_models}x{n_models}, {t} trials avg)")
        plt.xlabel("model index")
        plt.ylabel("model index")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r (avg)")
        plt.tight_layout()
        plt.savefig(png_c_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Further collapses: by rho (6x6) and by task loss (4x4)
        # Build grouping indices aligned with model order used above
        if len(kept_models_meta) != n_models:
            # Safety guard; if mismatch, skip further collapsing
            pass
        else:
            # Group indices by loss and by rho
            loss_to_indices: Dict[str, List[int]] = {}
            rho_to_indices: Dict[float, List[int]] = {}
            for idx, meta in enumerate(kept_models_meta):
                loss_to_indices.setdefault(meta["loss"], []).append(idx)
                rho_to_indices.setdefault(meta["rho"], []).append(idx)

            # Preserve appearance order for losses; rhos will follow first-appearance order (typically ascending)
            loss_labels: List[str] = []
            seen_losses = set()
            for meta in kept_models_meta:
                if meta["loss"] not in seen_losses:
                    loss_labels.append(meta["loss"])
                    seen_losses.add(meta["loss"])

            rho_values: List[float] = []
            seen_rhos = set()
            for meta in kept_models_meta:
                if meta["rho"] not in seen_rhos:
                    rho_values.append(meta["rho"])
                    seen_rhos.add(meta["rho"])

            # Collapse by rho (group by rho only) -> R x R
            R = len(rho_values)
            if R >= 2:
                rsa_by_rho = torch.empty((R, R), dtype=rsa_collapsed.dtype)
                for i, r_i in enumerate(rho_values):
                    idx_i = rho_to_indices[r_i]
                    rows_i = rsa_collapsed[idx_i, :]
                    for j, r_j in enumerate(rho_values):
                        idx_j = rho_to_indices[r_j]
                        block = rows_i[:, idx_j]
                        rsa_by_rho[i, j] = block.mean()

                # Save and plot
                rsa_rho_pt = os.path.join(models_root, f"{args.output_prefix}_collapsed_by_rho_{R}x{R}.pt")
                torch.save({
                    "rsa_matrix_by_rho": rsa_by_rho,
                    "rho_values": rho_values,
                    "original_models": [m["name"] for m in kept_models_meta],
                }, rsa_rho_pt)

                png_rho_path = os.path.join(models_root, f"{args.output_prefix}_collapsed_by_rho_{R}x{R}.png")
                plt.figure(figsize=(7, 6))
                im = plt.imshow(rsa_by_rho.numpy(), cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
                plt.title(f"RSA collapsed by rho ({R}x{R})")
                plt.xlabel("rho index")
                plt.ylabel("rho index")
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label("Pearson r (avg)")
                plt.tight_layout()
                plt.savefig(png_rho_path, dpi=200, bbox_inches="tight")
                plt.close()

            # Collapse by task loss (group by loss only) -> L x L
            L = len(loss_labels)
            if L >= 2:
                rsa_by_loss = torch.empty((L, L), dtype=rsa_collapsed.dtype)
                for i, loss_i in enumerate(loss_labels):
                    idx_i = loss_to_indices[loss_i]
                    rows_i = rsa_collapsed[idx_i, :]
                    for j, loss_j in enumerate(loss_labels):
                        idx_j = loss_to_indices[loss_j]
                        block = rows_i[:, idx_j]
                        rsa_by_loss[i, j] = block.mean()

                # Save and plot
                rsa_loss_pt = os.path.join(models_root, f"{args.output_prefix}_collapsed_by_loss_{L}x{L}.pt")
                torch.save({
                    "rsa_matrix_by_loss": rsa_by_loss,
                    "loss_labels": loss_labels,
                    "original_models": [m["name"] for m in kept_models_meta],
                }, rsa_loss_pt)

                png_loss_path = os.path.join(models_root, f"{args.output_prefix}_collapsed_by_loss_{L}x{L}.png")
                plt.figure(figsize=(7, 6))
                im = plt.imshow(rsa_by_loss.numpy(), cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
                plt.title(f"RSA collapsed by loss ({L}x{L})")
                plt.xlabel("loss index")
                plt.ylabel("loss index")
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label("Pearson r (avg)")
                plt.tight_layout()
                plt.savefig(png_loss_path, dpi=200, bbox_inches="tight")
                plt.close()
    else:
        # If not divisible, skip quietly per the "keep it simple" request
        pass


if __name__ == "__main__":
    main()
