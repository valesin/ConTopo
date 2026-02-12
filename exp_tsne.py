import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from utils.load import load_model_bundles, parse_model_load_args
from utils.experiments import get_cifar10_eval_loader, resolve_figure_path
from utils.train import load_cifar10_metadata

def main():
    args = parse_model_load_args()

    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
    )
    if not bundles:
        raise RuntimeError("No checkpoints found for the provided path.")

    # Eval-only CIFAR-10 loader
    val_loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    config = load_cifar10_metadata()
    class_names = config["CIFAR10_CLASSES"]
    torch.manual_seed(42)
    np.random.seed(42)

    target_samples = 2000
    for bundle in bundles:
        encoder = bundle.encoder
        meta = bundle.meta
        src = meta.get("figure_source") or meta.get("ckpt_path") or args.path
        stage = meta.get("stage")
        epoch = meta.get("encoder_epoch")
        print(f"Loaded stage: {stage} epoch: {epoch} from {src}")

        device = next(encoder.parameters()).device
        encoder.eval()

        feats, labs_all = [], []

        with torch.no_grad():
            for imgs, labs in val_loader:
                if len(labs_all) >= target_samples:
                    break
                imgs = imgs.to(device, non_blocking=True)
                out = encoder(imgs)
                if out.ndim > 2:
                    out = out.flatten(1)
                out = out.detach().cpu()

                remaining = target_samples - len(labs_all)
                if out.size(0) > remaining:
                    out = out[:remaining]
                    labs = labs[:remaining]
                feats.append(out)
                labs_all.extend(labs.tolist())

        if not feats:
            raise RuntimeError("No embeddings collected from the eval loader.")
        X = torch.cat(feats, dim=0).numpy()
        y = np.array(labs_all)

        tsne = TSNE(n_components=2, random_state=42)
        X2 = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
        for i in range(10):
            idx = (y == i)
            if idx.any():
                plt.scatter(
                    X2[idx, 0], X2[idx, 1],
                    c=colors[i],
                    label=class_names[i] if i < len(class_names) else f"class {i}"
                )

        plt.title('t-SNE Visualization of Feature Space (CIFAR-10 eval, 2000 samples)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        figurepath = resolve_figure_path(src, experiment="tsne")
        plt.savefig(figurepath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {figurepath}")

if __name__ == "__main__":
    main()
