
import random
import shutil
from importlib import import_module
from pathlib import Path

try:
    Image = import_module("PIL.Image")
except ModuleNotFoundError as error:
    raise RuntimeError(
        "Pillow is required. Install it with: python -m pip install Pillow"
    ) from error

# Reproducibility
random.seed(42)

# Dataset paths
RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset")

CLASSES = ["cat", "dog"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

for class_name in CLASSES:

    class_dir = RAW_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing dataset folder: {class_dir}"
        )

    images = []

    # Collect valid images
    for file in class_dir.iterdir():

        if file.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            with Image.open(file) as img:
                img.verify()

            images.append(file)

        except Exception:
            print("Skipping invalid image:", file)

    random.shuffle(images)

    total = len(images)

    if total < 10:
        raise ValueError(
            f"Not enough valid images for {class_name}"
        )

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():

        destination = OUTPUT_DIR / split_name / class_name

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in split_images:

            shutil.copy2(
                image_path,
                destination / image_path.name
            )

        print(
            f"{class_name} - {split_name}: "
            f"{len(split_images)} images"
        )

print("\nDataset splitting completed!")
