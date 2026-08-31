"""Download a manageable, clean WildFake/COCO evaluation subset.

The DALL-E input must be the actual WildFake DALL-E archive URL, copied from
the ModelScope Files page. For a ZIP archive, remotezip reads only selected
members using HTTP Range requests; it does not download the whole archive.

The COCO images are downloaded individually from the official COCO host.
The output manifest has labels real=0 and AI=1.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
import ssl
from pathlib import Path
from urllib.request import Request, urlopen


COCO_ANNOTATIONS = (
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE = "https://images.cocodataset.org/val2017/{name}"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def get_bytes(url: str, insecure: bool = False) -> bytes:
    request = Request(url, headers={"User-Agent": "wildfake-eval-subset/1.0"})

    # Colab sometimes reports a certificate hostname mismatch for COCO.
    context = ssl._create_unverified_context() if insecure else None

    with urlopen(request, timeout=120, context=context) as response:
        return response.read()


def coco_names(cache_dir: Path) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotation_file = cache_dir / "instances_val2017.json"
    if not annotation_file.exists():
        print("Downloading COCO annotations (~240 MB, one time)...")
        archive_data = get_bytes(COCO_ANNOTATIONS, insecure=True)
        with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
            annotation_file.write_bytes(
                archive.read("annotations/instances_val2017.json")
            )
    data = json.loads(annotation_file.read_text())
    return [item["file_name"] for item in data["images"]]


def download_coco(root: Path, count: int, seed: int) -> list[dict]:
    names = coco_names(root / "metadata")
    selected = random.Random(seed).sample(names, min(count, len(names)))
    output = root / "real"
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, name in enumerate(selected, 1):
        destination = output / name
        if not destination.exists():
            print(f"COCO {i}/{len(selected)}: {name}")
            destination.write_bytes(
                get_bytes(COCO_IMAGE.format(name=name), insecure=True)
            )
        rows.append({
            "image_path": str(destination.resolve()),
            "label": 0,
            "source": "WildFake_COCO_val2017",
            "condition": "clean",
        })
    return rows


def extract_dalle(root: Path, archive_url: str, count: int, seed: int) -> list[dict]:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise SystemExit("Install remotezip first: pip install remotezip") from exc

    output = root / "ai"
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Opening the remote ZIP directory (requires HTTP Range support)...")
    with RemoteZip(archive_url) as archive:
        names = [
            name for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in EXTENSIONS
        ]
        if not names:
            raise RuntimeError("No image members found in the DALL-E ZIP archive")

        selected = random.Random(seed).sample(names, min(count, len(names)))
        for i, member in enumerate(selected, 1):
            suffix = Path(member).suffix.lower() or ".jpg"
            destination = output / f"dalle_{i:05d}{suffix}"
            if not destination.exists():
                print(f"DALL-E {i}/{len(selected)}: {member}")
                destination.write_bytes(archive.read(member))
            rows.append({
                "image_path": str(destination.resolve()),
                "label": 1,
                "source": "WildFake_DALLE_Advanced",
                "condition": "clean",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dalle-url", required=True,
                        help="Direct URL to the WildFake DALL-E ZIP file")
    parser.add_argument("--output", type=Path, default=Path("wildfake_subset"))
    parser.add_argument("--real-count", type=int, default=1000)
    parser.add_argument("--ai-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    real_rows = download_coco(args.output, args.real_count, args.seed)
    ai_rows = extract_dalle(args.output, args.dalle_url, args.ai_count, args.seed)
    rows = real_rows + ai_rows
    random.Random(args.seed).shuffle(rows)

    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "source", "condition"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} clean images")
    print(f"Manifest: {manifest.resolve()}")
    print("label=0: real COCO; label=1: WildFake DALL-E")


if __name__ == "__main__":
    main()