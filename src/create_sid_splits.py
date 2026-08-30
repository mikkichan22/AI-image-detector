from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import random


SEED = 42
VALIDATION_RATIO = 0.15

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "SID_Set_subset"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "sid_splits.csv"
)


def get_file_hash(image_path):
    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    records = []

    for class_name, label in [
        ("REAL", 0),
        ("FAKE", 1),
    ]:
        folder = ROOT / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        for image_path in folder.glob("*.jpg"):
            relative_path = image_path.relative_to(
                PROJECT_ROOT
            )

            records.append(
                {
                    "image_path": str(relative_path),
                    "label": label,
                    "class_name": class_name,
                    "source_dataset": "SID_Set",
                    "file_hash": get_file_hash(image_path),
                }
            )

    duplicate_groups = defaultdict(list)

    for record in records:
        duplicate_groups[record["file_hash"]].append(
            record
        )

    print("Images found:", len(records))
    print(
        "Duplicate groups:",
        sum(
            len(group) > 1
            for group in duplicate_groups.values()
        ),
    )

    groups_by_label = {
        0: [],
        1: [],
    }

    for group in duplicate_groups.values():
        label = group[0]["label"]
        groups_by_label[label].append(group)

    rng = random.Random(SEED)
    final_records = []

    for label in [0, 1]:
        groups = groups_by_label[label]
        rng.shuffle(groups)

        validation_group_count = int(
            len(groups) * VALIDATION_RATIO
        )

        validation_groups = groups[
            :validation_group_count
        ]

        train_groups = groups[
            validation_group_count:
        ]

        for group in train_groups:
            for record in group:
                record["split"] = "train"
                final_records.append(record)

        for group in validation_groups:
            for record in group:
                record["split"] = "validation"
                final_records.append(record)

    for record in final_records:
        record.pop("file_hash", None)

    fieldnames = [
        "image_path",
        "label",
        "class_name",
        "source_dataset",
        "split",
    ]

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(final_records)

    print("Created:", OUTPUT)

    for split in ["train", "validation"]:
        for label in [0, 1]:
            count = sum(
                record["split"] == split
                and record["label"] == label
                for record in final_records
            )

            print(
                f"{split:10} "
                f"label={label}: {count}"
            )


if __name__ == "__main__":
    main()