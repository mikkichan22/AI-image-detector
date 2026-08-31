from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PREDICTIONS_FILE = Path("results/predictions.csv")
METRICS_FILE = Path("results/metrics.csv")
ERRORS_FILE = Path("results/error_analysis.csv")


def evaluate_group(group):
    labels = group["label"]
    predictions = group["prediction"]

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "transform": group["transform"].iloc[0],
        "images": len(group),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_FILE}. "
            "Ask Person 2 for predictions.csv."
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {
        "image_path",
        "label",
        "prediction",
        "ai_probability",
        "transform",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    metrics = []

    for transform_name, group in predictions.groupby("transform"):
        metrics.append(evaluate_group(group))

    metrics_df = pd.DataFrame(metrics)

    clean_rows = metrics_df[
        metrics_df["transform"] == "clean"
    ]

    if not clean_rows.empty:
        clean_f1 = clean_rows.iloc[0]["f1"]
        metrics_df["f1_drop_from_clean"] = (
            clean_f1 - metrics_df["f1"]
        )
    else:
        metrics_df["f1_drop_from_clean"] = None

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_FILE, index=False)

    predictions["error_type"] = "correct"

    predictions.loc[
        (predictions["label"] == 0)
        & (predictions["prediction"] == 1),
        "error_type",
    ] = "false_positive"

    predictions.loc[
        (predictions["label"] == 1)
        & (predictions["prediction"] == 0),
        "error_type",
    ] = "false_negative"

    predictions.loc[
        predictions["prediction"] != predictions["label"],
        "error_type",
    ] = predictions["error_type"]

    incorrect = predictions[
        predictions["prediction"] != predictions["label"]
    ].copy()

    incorrect["confidence"] = incorrect["ai_probability"].apply(
        lambda probability: max(probability, 1 - probability)
    )

    incorrect = incorrect.sort_values(
        "confidence",
        ascending=False,
    )

    incorrect.to_csv(ERRORS_FILE, index=False)

    print(f"Saved metrics to {METRICS_FILE}")
    print(f"Saved errors to {ERRORS_FILE}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()