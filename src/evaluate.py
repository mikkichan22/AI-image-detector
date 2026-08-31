import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DEFAULT_INPUT = Path("results/predictions.csv")
DEFAULT_METRICS = Path("results/metrics.csv")
DEFAULT_ERRORS = Path("results/error_analysis.csv")


def normalize_label(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"real", "0"}:
            return 0
        if cleaned in {"ai generated", "ai-generated", "ai_generated", "1"}:
            return 1
        return None

    if isinstance(value, (int, float, bool)):
        return int(value)

    return None


def normalize_prediction(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        return normalize_label(value)

    if isinstance(value, (int, float, bool)):
        return int(value)

    return None


def load_predictions(path):
    path = Path(path)

    if path.suffix.lower() == ".json":
        data = pd.read_json(path)
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported prediction file type: {path.suffix}")

    if "pred" in data.columns and "prediction" not in data.columns:
        data["prediction"] = data["pred"].map(normalize_prediction)

    if "label" in data.columns:
        data["label"] = data["label"].map(normalize_label)

    if "prediction" in data.columns:
        data["prediction"] = data["prediction"].map(normalize_prediction)

    if "transform" not in data.columns:
        data["transform"] = "clean"

    if "ai_probability" not in data.columns:
        if "confidence" in data.columns:
            data["ai_probability"] = data["confidence"]
        else:
            data["ai_probability"] = 0.0

    return data


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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prediction output from CSV or JSON files. "
            "JSON may use 'pred' with values like 'real' or 'AI generated'."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to predictions CSV or JSON file.",
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS),
        help="Where to save the per-transform metrics CSV.",
    )
    parser.add_argument(
        "--errors",
        default=str(DEFAULT_ERRORS),
        help="Where to save the error analysis CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    metrics_path = Path(args.metrics)
    errors_path = Path(args.errors)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {input_path}")

    predictions = load_predictions(input_path)

    required_columns = {"image_path", "label", "prediction", "ai_probability"}
    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if predictions["label"].isnull().any() or predictions["prediction"].isnull().any():
        bad_rows = predictions[
            predictions["label"].isnull() | predictions["prediction"].isnull()
        ]
        raise ValueError(
            "Could not interpret some label or prediction values. "
            "Expected labels like 0/1 or 'real'/'AI generated'.\n"
            f"Problem rows:\n{bad_rows.head()}"
        )

    metrics = []

    for transform_name, group in predictions.groupby("transform"):
        metrics.append(evaluate_group(group))

    metrics_df = pd.DataFrame(metrics)

    clean_rows = metrics_df[metrics_df["transform"] == "clean"]

    if not clean_rows.empty:
        clean_f1 = clean_rows.iloc[0]["f1"]
        metrics_df["f1_drop_from_clean"] = clean_f1 - metrics_df["f1"]
    else:
        metrics_df["f1_drop_from_clean"] = None

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    predictions["error_type"] = "correct"

    predictions.loc[
        (predictions["label"] == 0) & (predictions["prediction"] == 1),
        "error_type",
    ] = "false_positive"

    predictions.loc[
        (predictions["label"] == 1) & (predictions["prediction"] == 0),
        "error_type",
    ] = "false_negative"

    incorrect = predictions[predictions["prediction"] != predictions["label"]].copy()
    incorrect["confidence"] = incorrect["ai_probability"].apply(
        lambda probability: max(probability, 1 - probability)
    )
    incorrect = incorrect.sort_values("confidence", ascending=False)

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    incorrect.to_csv(errors_path, index=False)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved errors to {errors_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()