from collections import Counter
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "SID_Set_subset"
)

TARGET_PER_CLASS = 10_000

REAL_DIR = OUTPUT_ROOT / "REAL"
FAKE_DIR = OUTPUT_ROOT / "FAKE"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading SID_Set in streaming mode...")

    dataset = load_dataset(
        "saberzl/SID_Set",
        split="train",
        streaming=True,
    )

    counts = Counter()

    for row_number, row in enumerate(dataset):

        label = int(row["label"])

        # SID label 0 = real.
        # SID label 1 = full synthetic.
        # SID label 2 = tampered, excluded for now.
        if label not in [0, 1]:
            continue

        if counts[label] >= TARGET_PER_CLASS:
            continue

        image = row["image"].convert("RGB")

        if label == 0:
            output_dir = REAL_DIR
        else:
            output_dir = FAKE_DIR

        output_path = (
            output_dir
            / f"{label}_{counts[label]:06d}.jpg"
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95,
        )

        counts[label] += 1

        if sum(counts.values()) % 1000 == 0:
            print("Saved:", dict(counts))

        if (
            counts[0] >= TARGET_PER_CLASS
            and counts[1] >= TARGET_PER_CLASS
        ):
            break

    print("\nFinished.")
    print("Images saved:", dict(counts))
    print("Output:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()