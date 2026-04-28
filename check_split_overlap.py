import os
from torchvision import datasets

root = "./dataset"  # adjust if needed
train_ds = datasets.ImageFolder(root=os.path.join(root, "imagenet100/train"))
test_ds = datasets.ImageFolder(root=os.path.join(root, "imagenet100/val"))

# ── Directory-level overlap (train dir vs test dir) ──────────────────────────
train_files = {os.path.basename(p) for p, _ in train_ds.imgs}
test_files = {os.path.basename(p) for p, _ in test_ds.imgs}

dir_overlap = train_files & test_files
print(f"train dir images : {len(train_files)}")
print(f"test  dir images : {len(test_files)}")
print(f"filename overlap (train dir vs test dir) : {len(dir_overlap)}")
if dir_overlap:
    print("Sample overlapping filenames:", list(dir_overlap)[:5])

# ── Pipeline val split (first 50/class from train dir) vs test dir ───────────
val_per_class = 50
class_counts: dict[int, int] = {}
val_idx: list[int] = []
for idx, (_, label) in enumerate(train_ds.imgs):
    class_counts.setdefault(label, 0)
    if class_counts[label] < val_per_class:
        val_idx.append(idx)
        class_counts[label] += 1

val_files = {os.path.basename(train_ds.imgs[i][0]) for i in val_idx}
pipeline_overlap = val_files & test_files

print(f"\npipeline val images : {len(val_files)}")
print(f"pipeline val vs test overlap : {len(pipeline_overlap)}")
if pipeline_overlap:
    print("Overlapping filenames:", list(pipeline_overlap)[:10])
else:
    print("No overlap — pipeline val and test splits are disjoint.")
