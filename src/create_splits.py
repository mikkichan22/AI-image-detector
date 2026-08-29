from pathlib import Path
import csv
import random

SEED = 42
random.seed(SEED)

root = Path("data/raw/CIFAKE")
output = Path("data/splits.csv")
output.parent.mkdir(parents=True, exist_ok=True)

records = []

# Use CIFAKE's official train/test separation.
# Split official training data into train and validation.
for original_split in ["train", "test"]:
    for class_name, label in [("REAL", 0), ("FAKE", 1)]:
        folder = root / original_split / class_name

        for image_path in folder.glob("*"):
            if image_path.is_file():
                records.append({
                    "image_path": str(image_path),
                    "original_label": label,
                    "label": label,
                    "class_name": class_name,
                    "source_dataset": "CIFAKE",
                    "original_split": original_split,
                })

train_records = [
    record for record in records
    if record["original_split"] == "train"
]

test_records = [
    record for record in records
    if record["original_split"] == "test"
]

final_records = []

# Stratified 85/15 split of the official training data
for label in [0, 1]:
    class_records = [
        record for record in train_records
        if record["label"] == label
    ]

    random.shuffle(class_records)
    validation_count = int(len(class_records) * 0.15)

    validation_records = class_records[:validation_count]
    actual_train_records = class_records[validation_count:]

    for record in actual_train_records:
        record["split"] = "train"
        final_records.append(record)

    for record in validation_records:
        record["split"] = "validation"
        final_records.append(record)

# Keep official CIFAKE test data as the internal test set
for record in test_records:
    record["split"] = "test"
    final_records.append(record)

fieldnames = [
    "image_path",
    "original_label",
    "label",
    "class_name",
    "source_dataset",
    "original_split",
    "split",
]

with output.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_records)

print(f"Created {output}")
print(f"Total images: {len(final_records)}")

for split_name in ["train", "validation", "test"]:
    for label in [0, 1]:
        count = sum(
            1 for record in final_records
            if record["split"] == split_name
            and record["label"] == label
        )
        print(f"{split_name:10} label={label}: {count}")