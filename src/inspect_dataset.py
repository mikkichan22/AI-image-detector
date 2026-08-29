from pathlib import Path
from PIL import Image
from collections import Counter

root = Path("data/raw/CIFAKE")
dimensions = Counter()
formats = Counter()
corrupt = []

for path in root.rglob("*"):
    if path.is_file():
        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                dimensions[image.size] += 1
                formats[image.format] += 1
        except Exception:
            corrupt.append(str(path))

print("Image dimensions:")
for size, count in dimensions.most_common():
    print(size, count)

print("\nFile formats:")
for file_format, count in formats.most_common():
    print(file_format, count)

print("\nCorrupt images:", len(corrupt))
for path in corrupt[:20]:
    print(path)