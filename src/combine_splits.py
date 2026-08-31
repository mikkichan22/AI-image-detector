from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CIFAKE_SPLITS = (
    PROJECT_ROOT
    / "data"
    / "splits.csv"
)

SID_SPLITS = (
    PROJECT_ROOT
    / "data"
    / "sid_splits.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "sid_priority_splits.csv"
)

# Number of CIFAKE images per class used for training.
# SID remains the majority source.
CIFAKE_TRAIN_PER_CLASS = 2_500

# Number of CIFAKE validation images per class to retain.
# With the default value this keeps 1,000 REAL and 1,000 FAKE images.
CIFAKE_VALIDATION_PER_CLASS = 1_000

RANDOM_SEED = 42


def main():
    cifake = pd.read_csv(CIFAKE_SPLITS)
    sid = pd.read_csv(SID_SPLITS)

    cifake["source_dataset"] = "CIFAKE"
    sid["source_dataset"] = "SID_Set"

    cifake_train = cifake[
        cifake["split"] == "train"
    ]

    sampled_cifake_train = (
        cifake_train
        .groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(
                    len(group),
                    CIFAKE_TRAIN_PER_CLASS,
                ),
                random_state=RANDOM_SEED,
            )
        )
        .reset_index(drop=True)
    )

    cifake_non_train = cifake[
        cifake["split"] != "train"
    ]

    cifake_selected = pd.concat(
        [
            sampled_cifake_train,
            cifake_non_train,
        ],
        ignore_index=True,
    )

    combined = pd.concat(
        [
            cifake_selected,
            sid,
        ],
        ignore_index=True,
    )

    # Build the final validation policy in the same script:
    # - preserve every train and test row;
    # - preserve every SID validation row;
    # - add a balanced CIFAKE validation sample.
    non_validation = combined[
        combined["split"] != "validation"
    ].copy()

    sid_validation = combined[
        (combined["split"] == "validation")
        & (combined["source_dataset"] == "SID_Set")
    ].copy()

    cifake_validation = combined[
        (combined["split"] == "validation")
        & (combined["source_dataset"] == "CIFAKE")
    ]

    cifake_validation_sample = (
        cifake_validation
        .groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(
                    len(group),
                    CIFAKE_VALIDATION_PER_CLASS,
                ),
                random_state=RANDOM_SEED,
            )
        )
        .reset_index(drop=True)
    )

    sid_priority = pd.concat(
        [
            non_validation,
            sid_validation,
            cifake_validation_sample,
        ],
        ignore_index=True,
    )

    sid_priority.to_csv(OUTPUT, index=False)

    print("Created:", OUTPUT)

    print("\nFinal split counts:")

    print(
        sid_priority.groupby(
            [
                "source_dataset",
                "split",
                "label",
            ]
        ).size()
    )

    missing_paths = []

    for path_string in sid_priority["image_path"]:
        path = Path(path_string)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            missing_paths.append(str(path))

    print("Missing paths:", len(missing_paths))

    if missing_paths:
        print("\n".join(missing_paths[:10]))
        raise FileNotFoundError(
            "Some image paths do not exist."
        )


if __name__ == "__main__":
    main()