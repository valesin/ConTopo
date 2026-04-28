import os
from torchvision import datasets

root = "./dataset"  # adjust if needed
train_ds = datasets.ImageFolder(root=os.path.join(root, "imagenet100/train"))
val_ds = datasets.ImageFolder(root=os.path.join(root, "imagenet100/val"))

train_files = {os.path.basename(p) for p, _ in train_ds.imgs}
val_files = {os.path.basename(p) for p, _ in val_ds.imgs}

overlap = train_files & val_files
print(f"train images : {len(train_files)}")
print(f"val   images : {len(val_files)}")
print(f"filename overlap : {len(overlap)}")
if overlap:
    print("Sample overlapping filenames:", list(overlap)[:5])
else:
    print("No filename overlap — splits are disjoint.")
