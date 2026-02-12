import math

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

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
    # Collect one example per CIFAR-10 class (deterministic order)
    exemplar_imgs, exemplar_labels = [], []
    for imgs, labs in val_loader:
        for img, lab in zip(imgs, labs):
            c = int(lab)
            if c not in exemplar_labels:
                exemplar_imgs.append(img)
                exemplar_labels.append(c)
            if len(exemplar_imgs) == 10:
                break
        if len(exemplar_imgs) == 10:
            break

    if len(exemplar_imgs) < 10:
        raise RuntimeError(f"Only found {len(exemplar_imgs)} distinct classes; need 10.")

    sample_imgs_cpu = torch.stack(exemplar_imgs)  # (10, 3, 32, 32)
    n_images = sample_imgs_cpu.size(0)

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

    # undo dataset normalisation
    inv_norm = T.Normalize(
        mean=[-m / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
        std=[1 / s for s in CIFAR10_STD],
    )
    imgs_show = inv_norm(sample_imgs_cpu.clone()).clamp(0, 1)

    for bundle in bundles:
        encoder = bundle.encoder
        meta = bundle.meta
        src = meta.get("figure_source") or meta.get("ckpt_path") or args.path
        stage = meta.get("stage")
        epoch = meta.get("encoder_epoch")
        print(f"Loaded stage: {stage} epoch: {epoch} from {src}")

        device = next(encoder.parameters()).device
        encoder.eval()

        sample_imgs = sample_imgs_cpu.to(device, non_blocking=True)

        activations = []

        def hook_fn(module, inp, out):
            activations.append(out.detach().cpu().flatten(1))

        enc_module = encoder.module if hasattr(encoder, "module") else encoder
        if not hasattr(enc_module, "fc"):
            raise AttributeError("Encoder has no 'fc' module to hook.")
        handle = enc_module.fc.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = encoder(sample_imgs)
        handle.remove()

        if not activations:
            raise RuntimeError("Forward hook did not fire; check that 'fc' exists on the encoder.")
        act = activations[0]

        emb_dim = act.shape[1]
        h = int(math.sqrt(emb_dim))
        while emb_dim % h != 0:
            h -= 1
        w = emb_dim // h

        rows = n_images
        fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))

        v = act.abs().max().item()

        for i in range(rows):
            axes[i, 0].imshow(imgs_show[i].permute(1, 2, 0))
            axes[i, 0].set_title(f"{class_names[exemplar_labels[i]]}", fontsize=10)
            axes[i, 0].axis('off')

            heat = act[i].reshape(h, w)
            im = axes[i, 1].imshow(
                heat,
                cmap='bwr',
                interpolation='nearest',
                vmin=-v,
                vmax=v,
            )
            axes[i, 1].set_title("FC1 activations", fontsize=10)
            axes[i, 1].axis('off')
            fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        figurepath = resolve_figure_path(src, experiment="actmaps")
        plt.savefig(figurepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {figurepath}")


if __name__ == "__main__":
    main()
