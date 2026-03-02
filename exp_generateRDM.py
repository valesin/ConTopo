""" Run with
python run_all_experiments.py exp_generateRDM.py save/ResNet18/models/
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils.load import (
    parse_model_load_args,
    load_model_bundles,
)
from utils.experiments import (
    get_cifar10_eval_loader,
    select_deterministic_cifar10_subset,
    compute_embeddings,
    pearson_rdm,
    upper_triangle_vector,
)


def main():
    args = parse_model_load_args()

    # Load all encoders from the provided model folder (one per trial),
    # selecting the checkpoint combination indicated by --prefer (defaults to 'best').
    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )

    # Do not cap the number of trials here
    # if len(bundles) > 5:
    #     print(f"Found {len(bundles)} runs; keeping the first 5 (trials 0-4).")
    #     bundles = bundles[:5]

    if len(bundles) == 0:
        raise RuntimeError("No checkpoints found in the provided model folder.")

    # Eval-only CIFAR-10 loader (deterministic order)
    val_loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Deterministically select 100 samples per class (1000 total)
    samples_cpu, labels = select_deterministic_cifar10_subset(val_loader, per_class=100)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # For each encoder, compute embeddings on the fixed 1000 samples and then the RDM
    rdms = []
    metas = []
    cosine_results = []
    for idx, bundle in enumerate(bundles):
        encoder = bundle.encoder
        meta = bundle.meta
        device = next(encoder.parameters()).device
        feats = compute_embeddings(encoder, samples_cpu, device, args.batch_size)
        rdm = pearson_rdm(feats)
        rdms.append(rdm)
        try:
            run_folder = os.path.dirname(meta.get("ckpt_path", ""))
            run_name = os.path.basename(run_folder) if run_folder else f"trial_{idx:02d}"
        except Exception:
            run_name = f"trial_{idx:02d}"
        # Keep compact meta info for traceability
        meta_info = {
            "stage": meta.get("stage"),
            "encoder_epoch": meta.get("encoder_epoch"),
            "ckpt_path": meta.get("ckpt_path"),
            "classifier_ckpt": meta.get("classifier_ckpt"),
            "run_folder": meta.get("run_folder"),
            "run_name": run_name,
        }
        metas.append(meta_info)

        norm_feats = F.normalize(feats, p=2, dim=1)
        sim_matrix = norm_feats @ norm_feats.t()
        tri_i, tri_j = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
        pair_sims = sim_matrix[tri_i, tri_j].to(dtype=torch.float32)
        same_mask = labels_tensor[tri_i] == labels_tensor[tri_j]
        within_sims = pair_sims[same_mask].cpu()
        across_sims = pair_sims[~same_mask].cpu()
        cosine_results.append({
            "run_name": run_name,
            "within": within_sims,
            "across": across_sims,
            "meta": meta_info,
        })

    # Save all trial RDMs in a single file under the given model folder
    model_folder = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    base = os.path.basename(os.path.normpath(model_folder))
    out_name = f"RDM_{base}.pt"
    out_path = os.path.join(model_folder, out_name)

    # Compute averaged RDM across trials (full matrix for plotting), then compress for saving
    avg_rdm = torch.stack(rdms, dim=0).mean(dim=0) if len(rdms) > 0 else None

    # Save per-trial figures (full RDMs for visualization)
    for idx, (rdm, meta) in enumerate(zip(rdms, metas)):
        run_name = meta.get("run_name") or f"trial_{idx:02d}"
        fig_name = f"RDM_{base}__{run_name}.png"
        fig_path = os.path.join(model_folder, fig_name)

        plt.figure(figsize=(8, 8))
        im = plt.imshow(rdm.numpy(), cmap="viridis", interpolation="nearest")
        plt.title(f"RDM: {run_name}")
        plt.xlabel("Samples (N=1000)")
        plt.ylabel("Samples (N=1000)")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="1 - Pearson r")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved RDM figure: {fig_path}")

    for stats in cosine_results:
        run_name = stats["run_name"]
        cos_pt_name = f"CosineSims_{base}__{run_name}.pt"
        cos_pt_path = os.path.join(model_folder, cos_pt_name)
        torch.save({
            "within": stats["within"],
            "across": stats["across"],
            "run_name": run_name,
            "meta": stats["meta"],
            "labels": labels,
        }, cos_pt_path)
        within_np = stats["within"].numpy()
        across_np = stats["across"].numpy()
        within_mean = float(within_np.mean())
        within_std = float(within_np.std())
        across_mean = float(across_np.mean())
        across_std = float(across_np.std())
        print(f"{run_name} - Within class cosine similarity: mean={within_mean:.4f}, std={within_std:.4f}")
        print(f"{run_name} - Across class cosine similarity: mean={across_mean:.4f}, std={across_std:.4f}")
        edges = torch.linspace(-1.0, 1.0, steps=60).numpy()
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.hist(across_np, bins=edges, color="#d95f02", alpha=0.65, density=True, label="Across class")
        ax.hist(within_np, bins=edges, color="#1b9e77", alpha=0.65, density=True, label="Within class")
        ax.set_title(f"Cosine similarity: {run_name}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        ax.grid(False)
        plt.tight_layout()
        cos_fig_name = f"CosineSims_{base}__{run_name}.png"
        cos_fig_path = os.path.join(model_folder, cos_fig_name)
        plt.savefig(cos_fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved cosine similarity data: {cos_pt_path}")
        print(f"Saved cosine similarity figure: {cos_fig_path}")

    # Save averaged RDM figure if available
    if avg_rdm is not None:
        avg_fig_name = f"AvgRDM_{base}.png"
        avg_fig_path = os.path.join(model_folder, avg_fig_name)
        plt.figure(figsize=(8, 8))
        im = plt.imshow(avg_rdm.numpy(), cmap="viridis", interpolation="nearest")
        plt.title("Average RDM across trials")
        plt.xlabel("Samples (N=1000)")
        plt.ylabel("Samples (N=1000)")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="1 - Pearson r")
        plt.tight_layout()
        plt.savefig(avg_fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved Avg RDM figure: {avg_fig_path}")

    # Compress RDMs to upper-triangular vectors for saving (exclude diagonal to avoid trivial zeros)
    rdms_upper = [
        upper_triangle_vector(rdm, include_diagonal=False) for rdm in rdms
    ]
    avg_rdm_upper = upper_triangle_vector(avg_rdm, include_diagonal=False) if avg_rdm is not None else None

    # Second-level analysis: correlate RDMs across trials (unique pairs, exclude self)
    # Use upper-triangle vectors WITHOUT the diagonal for correlation to avoid trivial zeros.
    if len(rdms) >= 2:
        vecs_no_diag = [
            upper_triangle_vector(rdm, include_diagonal=False) for rdm in rdms
        ]
        corrs = []
        eps = 1e-8
        for i in range(len(vecs_no_diag)):
            xi = vecs_no_diag[i]
            xi_c = xi - xi.mean()
            xi_n = xi_c.norm().clamp_min(eps)
            for j in range(i + 1, len(vecs_no_diag)):
                xj = vecs_no_diag[j]
                xj_c = xj - xj.mean()
                xj_n = xj_c.norm().clamp_min(eps)
                r = float((xi_c @ xj_c) / (xi_n * xj_n))
                corrs.append(r)
        stats_path = os.path.join(model_folder, f"RDMConsistency_{base}.pt")
        if corrs:
            import math as _m
            mean_corr = sum(corrs) / len(corrs)
            if len(corrs) > 1:
                var = sum((c - mean_corr) ** 2 for c in corrs) / (len(corrs) - 1)
                std_corr = _m.sqrt(var)
            else:
                std_corr = 0.0
            torch.save({
                "mean": float(mean_corr),
                "std": float(std_corr),
                "num_pairs": len(corrs),
                "num_trials": len(rdms),
                "note": "Pairwise Pearson correlation of upper-triangle (no diagonal)."
            }, stats_path)
        else:
            torch.save({
                "num_pairs": 0,
                "num_trials": len(rdms),
                "message": "Insufficient pairs to compute correlation."
            }, stats_path)
        print(f"Saved RDM consistency stats to: {stats_path}")
    else:
        print("RDM consistency across trials — need at least 2 trials. Skipping stats save.")

    # Save all trial RDMs (upper triangle only)
    payload = {
        "rdms_upper": rdms_upper,   # list of length n; each is 1D tensor (upper triangle incl. diag)
        "N": 1000,                  # original matrix size
        "include_diagonal": False,
        "labels": labels,           # list[int] length 1000
        "metas": metas,             # minimal metadata per trial
        "model_folder": model_folder,
        "prefer": args.prefer,
        "cosine_similarities": [
            {
                "run_name": stats["run_name"],
                "within": stats["within"],
                "across": stats["across"],
            }
            for stats in cosine_results
        ],
    }
    torch.save(payload, out_path)
    print(f"Saved {len(rdms_upper)} upper-triangular RDMs to: {out_path}")

    # Save averaged RDM (upper triangle only) in a separate file
    if avg_rdm_upper is not None:
        avg_name = f"AvgRDM_{base}.pt"
        avg_path = os.path.join(model_folder, avg_name)
        avg_payload = {
            "avg_rdm_upper": avg_rdm_upper,
            "N": 1000,
            "include_diagonal": False,
            "labels": labels,
            "num_trials": len(rdms_upper),
            "model_folder": model_folder,
            "prefer": args.prefer,
        }
        torch.save(avg_payload, avg_path)
        print(f"Saved averaged upper-triangular RDM to: {avg_path}")

    # (Figures saved above.)


if __name__ == "__main__":
    main()
