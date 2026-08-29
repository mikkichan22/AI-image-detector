from pathlib import Path
import csv
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SPLITS_FILE = Path("data/splits.csv")
OUTPUT_ROOT = Path("data/robustness")
MANIFEST_FILE = Path("data/robustness_manifest.csv")

TRANSFORMS = [
    ("clean", "none"),
    ("jpeg_70", "70"),
    ("jpeg_30", "30"),
    ("blur_1.0", "1.0"),
    ("resize_0.5", "0.5"),
    ("noise_0.05", "0.05"),
    ("colour_jitter", "20%"),
    ("crop_80", "80%"),
]


def apply_transform(image, transform_name):
    image = image.convert("RGB")

    if transform_name == "clean":
        return image

    if transform_name == "jpeg_70":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if transform_name == "jpeg_30":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=30)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if transform_name == "blur_1.0":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))

    if transform_name == "resize_0.5":
        width, height = image.size
        smaller = image.resize((width // 2, height // 2))
        return smaller.resize((width, height))

    if transform_name == "noise_0.05":
        array = np.asarray(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, 0.05, array.shape)
        noisy = np.clip(array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))

    if transform_name == "colour_jitter":
        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(1.2)
        image = ImageEnhance.Color(image).enhance(1.2)
        return image

    if transform_name == "crop_80":
        width, height = image.size
        new_width = int(width * 0.8)
        new_height = int(height * 0.8)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height))

    raise ValueError(f"Unknown transform: {transform_name}")


def main():
    with SPLITS_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    test_rows = [row for row in rows if row["split"] == "test"]

    manifest_rows = []

    for index, row in enumerate(test_rows, start=1):
        source_path = Path(row["image_path"])

        try:
            image = Image.open(source_path).convert("RGB")
        except Exception as error:
            print(f"Skipping {source_path}: {error}")
            continue

        filename = f"{index:05d}.jpg"

        for transform_name, parameter in TRANSFORMS:
            output_dir = OUTPUT_ROOT / transform_name
            output_dir.mkdir(parents=True, exist_ok=True)

            transformed = apply_transform(image, transform_name)
            output_path = output_dir / filename
            transformed.save(output_path, format="JPEG", quality=95)

            manifest_rows.append({
                "original_path": str(source_path),
                "transformed_path": str(output_path),
                "label": row["label"],
                "transform": transform_name,
                "parameter": parameter,
            })

        if index % 1000 == 0:
            print(f"Processed {index}/{len(test_rows)} test images")

    with MANIFEST_FILE.open("w", newline="") as file:
        fieldnames = [
            "original_path",
            "transformed_path",
            "label",
            "transform",
            "parameter",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Created {len(manifest_rows)} transformed images")
    print(f"Manifest saved to {MANIFEST_FILE}")


if __name__ == "__main__":
    main()