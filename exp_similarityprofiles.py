import os
import torch
import torch.nn.functional as F
from datetime import datetime, timezone

from utils import env
from utils.load import load_model_bundles
from utils.names import parse_run_name
from utils.experiments import (
    compute_embeddings,
    select_deterministic_cifar10_train_anchors,
)
import utils.ensemble_utils as ensemble_utils

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ANCHORS_PER_CLASS = 100
NUM_CLASSES = 10


def main():
    config_path = ensemble_utils.get_ensemble_config_path_from_cli()

    # ── 1. Load shared train anchor images (deterministic, CIFAR-10 TRAIN split) ──
    #    Mirrors the pattern in exp_generateRDM.py / select_deterministic_cifar10_subset:
    #    deterministic first-N-per-class selection, class-grouped ordering,
    #    eval-only transforms (no augmentation).
    print("Loading train anchor images...")
    anchor_images, anchor_labels, anchor_indices = (
        select_deterministic_cifar10_train_anchors(
            per_class=NUM_ANCHORS_PER_CLASS, root=env.DATA_ROOT
        )
    )
    total_anchors = NUM_CLASSES * NUM_ANCHORS_PER_CLASS
    assert anchor_images.shape[0] == total_anchors, (
        f"Expected {total_anchors} anchor images, got {anchor_images.shape[0]}"
    )
    assert anchor_labels.shape[0] == total_anchors
    print(f"Selected {total_anchors} train anchors ({NUM_ANCHORS_PER_CLASS} per class)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ens in ensemble_utils.iter_ensemble_inference_data_from_config(config_path):
        ensemble_name = ens["ensemble_name"]
        run_names = ens["run_names"]
        M = len(run_names)
        print(f"\nProcessing ensemble: {ensemble_name} ({M} models)")

        cosine_results = []
        labels = None

        with torch.no_grad():
            for i, run_name in enumerate(run_names):
                inference_data = ens["inference_data"][run_name]
                if i == 0:
                    labels = inference_data['labels']

                test_embeddings = inference_data['embeddings'].to(dtype=torch.float32)
                N_test = test_embeddings.shape[0]
                D = test_embeddings.shape[1]

                # Load the SAME encoder that produced the cached test embeddings
                # (load_model_bundles with prefer="best" matches get_or_run_inference)
                model_dir, trial = parse_run_name(run_name)
                trial_dir = os.path.join(env.MODELS_ROOT, model_dir, trial)
                bundles = load_model_bundles(
                    trial_dir, prefer="best", device=device, eval_mode=True
                )
                encoder = bundles[0].encoder

                # Compute anchor embeddings from shared train images using model k's encoder
                anchor_emb = compute_embeddings(
                    encoder, anchor_images, device, batch_size=256
                )
                # anchor_emb: [NUM_CLASSES * NUM_ANCHORS_PER_CLASS, D_enc]
                D_enc = anchor_emb.shape[1]
                assert D_enc == D, (
                    f"Anchor embedding dim ({D_enc}) != test embedding dim ({D}) "
                    f"for {run_name}"
                )

                # Reshape to [C, A, D] (images are in class-grouped order)
                anchors_3d = anchor_emb.view(NUM_CLASSES, NUM_ANCHORS_PER_CLASS, D)

                # ℓ2-normalise along the embedding dimension
                #   anchors: dim=2 (last) because shape is [C, A, D]
                #   test:    dim=1 (last) because shape is [N, D]
                anchors_norm = F.normalize(anchors_3d, p=2, dim=2)  # [C, A, D]
                test_norm    = F.normalize(test_embeddings, p=2, dim=1)  # [N, D]

                assert anchors_norm.shape == (NUM_CLASSES, NUM_ANCHORS_PER_CLASS, D)
                assert test_norm.shape == (N_test, D)

                # Cosine similarity: contract over embedding dim d → [C, A, N]
                sims = torch.einsum('cad,td->cat', anchors_norm, test_norm)
                mean_sims = sims.mean(dim=1)  # [C, N]
                assert mean_sims.shape == (NUM_CLASSES, N_test)

                cosine_results.append(mean_sims)

                # Free encoder memory before loading the next one
                del encoder, bundles, anchor_emb, anchors_3d, anchors_norm, sims
                torch.cuda.empty_cache()

                print(f"  [{i+1}/{M}] Computed similarity profile for {run_name}")

        cosine_results = torch.stack(cosine_results)  # [M, C, N]
        N = cosine_results.shape[2]
        assert cosine_results.shape == (M, NUM_CLASSES, N)

        # ── 2. Compute per-image RDMs across models ─────────────────────────
        def calc_triu_rdm(sim_across_models, correct_label=None, remove_correct_label=True):
            """
            sim_across_models: [M, C] similarity profiles for one image.
            Returns upper-triangular correlation-distance vector [M*(M-1)//2].
            """
            if remove_correct_label and correct_label is not None:
                sim_no_target = torch.cat(
                    [sim_across_models[:, :correct_label],
                     sim_across_models[:, correct_label + 1:]],
                    dim=1,
                )
                corr = torch.corrcoef(sim_no_target)
            else:
                corr = torch.corrcoef(sim_across_models)

            rdm = 1 - corr  # [M, M]

            # NaN guard: constant profiles → undefined Pearson → NaN
            if torch.isnan(rdm).any():
                nan_count = int(torch.isnan(rdm).sum().item())
                print(
                    f"    Warning: {nan_count} NaN(s) in RDM "
                    f"(constant profile), replacing with 0.0"
                )
                rdm = torch.nan_to_num(rdm, nan=0.0)

            K = rdm.shape[0]
            iu = torch.triu_indices(K, K, offset=1, device=rdm.device)
            return rdm[iu[0], iu[1]]  # [K*(K-1)//2]

        rdm_mats_remove = []
        rdm_mats_all = []
        with torch.no_grad():
            for img_idx in range(N):
                correct_label = labels[img_idx].item() if labels is not None else None
                rdm_remove = calc_triu_rdm(
                    cosine_results[:, :, img_idx],
                    correct_label=correct_label, remove_correct_label=True,
                )
                rdm_all = calc_triu_rdm(
                    cosine_results[:, :, img_idx],
                    correct_label=correct_label, remove_correct_label=False,
                )
                rdm_mats_remove.append(rdm_remove)
                rdm_mats_all.append(rdm_all)

        rdm_mats_remove = torch.stack(rdm_mats_remove)  # [N, M*(M-1)//2]
        rdm_mats_all = torch.stack(rdm_mats_all)

        # ── 3. Save with full metadata ──────────────────────────────────────
        ensemble_dir = ensemble_utils.get_ensemble_path_by_name(
            ensemble_name, save_dir=env.ENSEMBLES_ROOT
        )
        save_path = os.path.join(ensemble_dir, "similarity_profiles.pt")
        torch.save({
            "cosine_results": cosine_results,
            "rdm_mats_remove": rdm_mats_remove,
            "rdm_mats_all": rdm_mats_all,
            # ── Metadata for reproducibility & alignment verification ──
            "run_names": run_names,
            "num_anchors_per_class": NUM_ANCHORS_PER_CLASS,
            "num_classes": NUM_CLASSES,
            "anchor_source": "cifar10_train",
            "anchor_dataset_indices": anchor_indices,
            "anchor_labels": anchor_labels,
            "test_labels": labels,
            "num_models": M,
            "num_test_images": N,
            "created": datetime.now(timezone.utc).isoformat(),
        }, save_path)
        print(f"Saved similarity profiles to {save_path}")

def main_trial():
    # args = parse_model_load_args()

    num_anchors_per_class = 2

    # # Load all inference results for each trial in the model folder
    # model_dir = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    # bundles = get_inference_title_bundles(model_dir)

    # if len(bundles) == 0:
    #     raise RuntimeError("No cached inference files found in the provided model folder.")

    # For each trial, use cached embeddings and labels
    # rdms = []
    # metas = []
    cosine_results = []
    torch.manual_seed(0)
    tensor1 = torch.randint(low=0, high=4, size=(3, 2, 5)).float()
    tensor2 = torch.randint(low=0, high=4, size=(4, 5)).float()
    torch.manual_seed(1)
    tensor3 = torch.randint(low=0, high=4, size=(3, 2, 5)).float()
    tensor4 = torch.randint(low=0, high=4, size=(4, 5)).float()
    data = [(tensor1,tensor2),(tensor3,tensor4)]
    for idx, inference_data in enumerate(data):

        print(idx, "-" * 80)

        # Load anchors in a 3d tensor [#classes, #anchors_per_class, #dimensions]
        anchors_cpu, test_embeddings = inference_data

        print(anchors_cpu)
        print(test_embeddings)

        print("=" * 80)

        anchors_norm = F.normalize(anchors_cpu, p=2, dim=2)  # normalise along embedding dim
        test_norm = F.normalize(test_embeddings, p=2, dim=1)  # normalise along embedding dim
        print(anchors_norm)
        print(test_norm)
        sims = torch.einsum('cad,td->cat', anchors_norm, test_norm)

        print(sims)
        mean_sims = sims.mean(dim=1)
        print(mean_sims)

        cosine_results.append(mean_sims)
        t = 0
        print(mean_sims[:,t])

    print(cosine_results)


if __name__ == "__main__":
    main()