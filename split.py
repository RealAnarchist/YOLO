import random
import shutil
from pathlib import Path

root = Path("dataset")
images = sorted(p for p in (root / "images").iterdir() if p.is_file())

random.seed(0)
random.shuffle(images)
n_val = max(1, int(len(images) * 0.2))
val_images = images[:n_val]

for split in ["train", "val"]:
    (root / "images" / split).mkdir(exist_ok=True)
    (root / "labels" / split).mkdir(exist_ok=True)

for img in images:
    split = "val" if img in val_images else "train"
    label = root / "labels" / (img.stem + ".txt")
    shutil.move(str(img), str(root / "images" / split / img.name))
    if label.exists():
        shutil.move(str(label), str(root / "labels" / split / label.name))

print("train:", len(images) - n_val, "val:", n_val)