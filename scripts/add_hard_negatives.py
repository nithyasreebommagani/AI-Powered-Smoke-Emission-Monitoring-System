from pathlib import Path
import shutil
import random

source_folder = Path(r"D:\data set tuning")

dataset = Path(r"D:\combined_smoke_dataset")

train_img = dataset / "train" / "images"
train_lbl = dataset / "train" / "labels"

val_img = dataset / "valid" / "images"
val_lbl = dataset / "valid" / "labels"

images = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    images.extend(source_folder.glob(ext))

random.shuffle(images)

split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

for img in train_images:
    shutil.copy2(img, train_img / img.name)

    label = train_lbl / f"{img.stem}.txt"

    if not label.exists():
        label.write_text("")

for img in val_images:
    shutil.copy2(img, val_img / img.name)

    label = val_lbl / f"{img.stem}.txt"

    if not label.exists():
        label.write_text("")

print(f"Train negatives: {len(train_images)}")
print(f"Val negatives: {len(val_images)}")