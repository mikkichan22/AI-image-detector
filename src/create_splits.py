from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import random


SEED = 42
VALIDATION_RATIO = 0.15

random.seed(SEED)

ROOT = Path("data/raw/CIFAKE")
OUTPUT = Path("data/splits.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def get_file_hash(image_path):
    """Return a unique hash for an image file."""
    hash_function = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_function.update(chunk)

    return hash_function.hexdigest()


records = []

# Read the official CIFAKE train and test folders.
for original_split in ["train", "test"]:
    for class_name, label in [("REAL", 0), ("FAKE", 1)]:
        folder = ROOT / original_split / class_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        for image_path in folder.iterdir():
            if image_path.is_file():
                records.append(
                    {
                        "image_path": str(image_path),
                        "original_label": label,
                        "label": label,
                        "class_name": class_name,
                        "source_dataset": "CIFAKE",
                        "original_split": original_split,
                        "file_hash": get_file_hash(image_path),
                    }
                )

print(f"Found {len(records)} images.")


# Group identical images together using their file hash.
duplicate_groups = defaultdict(list)

for record in records:
    duplicate_groups[record["file_hash"]].append(record)


duplicate_group_count = sum(
    1 for group in duplicate_groups.values()
    if len(group) > 1
)

print(f"Found {duplicate_group_count} duplicate groups.")


# Assign every duplicate group to exactly one split.
#
# If a duplicate appears in the official test set,
# keep the entire group in test.
#
# Otherwise, split the official training groups into:
# 85% train and 15% validation.
final_records = []

training_groups_by_label = {
    0: [],
    1: [],
}

for file_hash, group in duplicate_groups.items():

    if any(record["original_split"] == "test" for record in group):
        chosen_split = "test"

        for record in group:
            record["split"] = chosen_split
            final_records.append(record)

    else:
        label = group[0]["label"]
        training_groups_by_label[label].append(group)


# Split training duplicate groups separately for REAL and FAKE.
for label in [0, 1]:
    groups = training_groups_by_label[label]

    random.shuffle(groups)

    validation_group_count = int(
        len(groups) * VALIDATION_RATIO
    )

    validation_groups = groups[:validation_group_count]
    train_groups = groups[validation_group_count:]

    for group in train_groups:
        for record in group:
            record["split"] = "train"
            final_records.append(record)

    for group in validation_groups:
        for record in group:
            record["split"] = "validation"
            final_records.append(record)


# Remove the internal hash before saving the CSV.
for record in final_records:
    record.pop("file_hash", None)


fieldnames = [
    "image_path",
    "original_label",
    "label",
    "class_name",
    "source_dataset",
    "original_split",
    "split",
]


with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_records)


print(f"\nCreated: {OUTPUT}")
print(f"Total images: {len(final_records)}")


print("\nFinal split counts:")

for split_name in ["train", "validation", "test"]:
    for label in [0, 1]:
        count = sum(
            1
            for record in final_records
            if record["split"] == split_name
            and record["label"] == label
        )

        class_name = "REAL" if label == 0 else "FAKE"

        print(
            f"{split_name:10} "
            f"{class_name:5} "
            f"label={label}: {count}"
        )