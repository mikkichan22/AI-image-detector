from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import csv

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOTS = [
    PROJECT_ROOT / "data" / "raw" / "CIFAKE",
    PROJECT_ROOT / "data" / "raw" / "SID_Set_subset",
]

SPLITS_FILE = PROJECT_ROOT / "data" / "sid_priority_splits.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

dimensions = Counter()
formats = Counter()
extensions = Counter()
split_counts = Counter()
class_counts = Counter()

corrupt = []
duplicate_groups = defaultdict(list)
image_records = []


def resolve_image_path(path_string):
    """Resolve paths in sid_priority_splits.csv across Colab locations."""
    raw_path = Path(str(path_string))

    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)

    # If the manifest was generated in another environment, reconstruct the
    # path below the known dataset directory name.
    for dataset_name in ["CIFAKE", "SID_Set_subset"]:
        parts = raw_path.parts
        if dataset_name in parts:
            suffix = Path(*parts[parts.index(dataset_name):])
            candidates.append(PROJECT_ROOT / "data" / "raw" / suffix)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]

# to detect duplicate pictures
def file_hash(path):
    digest = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# 1. Inspect actual image files from both datasets
available_roots = [root for root in DATA_ROOTS if root.exists()]

if not available_roots:
    raise FileNotFoundError(
        "Neither dataset directory exists: "
        + ", ".join(str(root) for root in DATA_ROOTS)
    )

for root in available_roots:
  for path in root.rglob("*"):
      if not path.is_file():
          continue

      if path.suffix.lower() not in VALID_EXTENSIONS:
          continue

      extensions[path.suffix.lower()] += 1

      try:
          with Image.open(path) as image:
              image.verify()

          with Image.open(path) as image:
              dimensions[image.size] += 1
              formats[image.format] += 1

          duplicate_groups[file_hash(path)].append(str(path))

      except Exception as error:
          corrupt.append((str(path), str(error)))


# 2. Inspect sid_priority_splits.csv
with SPLITS_FILE.open(newline="") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    split_counts[row["split"]] += 1
    class_name = row.get("class_name", row["label"])
    class_counts[(row["split"], row["label"], class_name)] += 1

    image_path = resolve_image_path(row["image_path"])

    image_records.append({
        "path": str(image_path),
        "split": row["split"],
        "label": row["label"],
    })


# 3. Print summary
print("Dataset roots:")
for root in available_roots:
    print(" -", root)

print("Manifest:", SPLITS_FILE)
print("Total image files:", sum(extensions.values()))

print("\nImage dimensions:")
for size, count in dimensions.most_common():
    print(f"{size}: {count}")

print("\nFile formats:")
for image_format, count in formats.most_common():
    print(f"{image_format}: {count}")

print("\nFile extensions:")
for extension, count in extensions.most_common():
    print(f"{extension}: {count}")

print("\nSplit counts:")
for split, count in split_counts.items():
    print(f"{split}: {count}")

print("\nClass counts:")
for key, count in sorted(class_counts.items()):
    print(f"{key}: {count}")

print("\nCorrupt images:", len(corrupt))
for path, error in corrupt[:20]:
    print(path, error)


# 4. Check missing CSV paths
missing_paths = [
    record["path"]
    for record in image_records
    if not Path(record["path"]).exists()
]

print("\nMissing paths in sid_priority_splits.csv:", len(missing_paths))
for path in missing_paths[:20]:
    print(path)


# 5. Check duplicate files
duplicate_groups = {
    digest: paths
    for digest, paths in duplicate_groups.items()
    if len(paths) > 1
}

print("\nDuplicate groups:", len(duplicate_groups))

for paths in list(duplicate_groups.values())[:10]:
    print(paths)


# 6. Check whether duplicates cross splits
path_to_split = {
    record["path"]: record["split"]
    for record in image_records
}

cross_split_duplicates = []

for paths in duplicate_groups.values():
    splits = {path_to_split[path] for path in paths if path in path_to_split}

    if len(splits) > 1:
        cross_split_duplicates.append((paths, splits))

print("\nDuplicate groups crossing splits:", len(cross_split_duplicates))

for paths, splits in cross_split_duplicates[:10]:
    print("Splits:", splits)
    print(paths)
