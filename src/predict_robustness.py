from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models
from torchvision.models import ResNet18_Weights

try:
    from src.create_robustness_sets import TRANSFORMS, apply_transform
    from src.transforms import evaluation_transform
except ModuleNotFoundError:
    from create_robustness_sets import TRANSFORMS, apply_transform
    from transforms import evaluation_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TRANSFORMS = [name for name, _ in TRANSFORMS]


def build_model():
    """
    Build the same ResNet-18 architecture used
    for the trained checkpoint.
    """

    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        2,
    )

    return model


def load_model(checkpoint_path, device):
    """
    Load the trained model checkpoint.
    Assumes the checkpoint contains:
        checkpoint["model_state_dict"]
    """

    model = build_model()

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown")
    )

    print(
        "Checkpoint validation F1:",
        checkpoint.get(
            "validation_f1",
            "unknown"
        )
    )

    model = model.to(device)

    # Important: evaluation mode
    model.eval()

    return model


def predict_image(
    model,
    image_path,
    device,
):
    """
    Predict one image.
 
    Returns:
        prediction:
            0 = real
            1 = AI
 
        ai_probability:
            probability of class 1
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    # IMPORTANT:
    # Use evaluation_transform, NOT augmented_transform.
    #
    # The robustness images have already been transformed
    # by create_robustness_sets.py.
    image_tensor = evaluation_transform(
        image
    )

    # Add batch dimension:
    # [3, 224, 224]
    # becomes
    # [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        prediction = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        # Assumes:
        # class 0 = real
        # class 1 = AI
        ai_probability = probabilities[
            0,
            1,
        ].item()

    return (
        prediction,
        ai_probability,
    )


def find_images(input_dir):
    """
    Find all supported image files in the input directory.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return sorted(images)


def build_manifest_from_dir(input_dir, transform_names, temp_dir):
    """
    Create a robustness-like manifest from a folder of raw images by applying
    the same transforms used by create_robustness_sets.py.
    """
    input_dir = Path(input_dir)
    temp_dir = Path(temp_dir)

    available_names = {name for name, _ in TRANSFORMS}
    invalid = [name for name in transform_names if name not in available_names]
    if invalid:
        raise ValueError(
            "Unknown transform names: "
            f"{invalid}. Available: {sorted(available_names)}"
        )

    manifest_rows = []
    image_paths = find_images(input_dir)

    print(f"Found {len(image_paths)} images in {input_dir}")

    for image_path in image_paths:
        raw_image = Image.open(image_path).convert("RGB")

        for transform_name in transform_names:
            transformed = apply_transform(raw_image, transform_name)
            transform_dir = temp_dir / transform_name
            transform_dir.mkdir(parents=True, exist_ok=True)

            output_name = f"{image_path.stem}_{transform_name}{image_path.suffix or '.jpg'}"
            output_path = transform_dir / output_name
            transformed.save(output_path, format="JPEG", quality=95)

            manifest_rows.append({
                "original_path": str(image_path),
                "transformed_path": str(output_path),
                "label": -1,
                "transform": transform_name,
                "parameter": next(
                    (param for name, param in TRANSFORMS if name == transform_name),
                    "unknown",
                ),
            })

    manifest = pd.DataFrame(manifest_rows)
    return manifest


def main(args):

    checkpoint_path = Path(
        args.checkpoint
    )

    output_path = Path(
        args.output
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Robustness manifest not found:\n"
                f"{manifest_path}"
            )

        print(f"Loading manifest:\n{manifest_path}")
        manifest = pd.read_csv(manifest_path)
        print(f"Found {len(manifest)} robustness images")

        required_columns = {"transformed_path", "label", "transform"}
        missing_columns = required_columns - set(manifest.columns)

        if missing_columns:
            raise ValueError(
                "Missing required columns in robustness manifest:\n"
                f"{sorted(missing_columns)}"
            )

    elif args.input_dir:
        transform_names = args.transforms or DEFAULT_TRANSFORMS
        temp_dir = Path(args.temp_dir) if args.temp_dir else Path("data/robustness_examples")
        manifest = build_manifest_from_dir(
            input_dir=args.input_dir,
            transform_names=transform_names,
            temp_dir=temp_dir,
        )
        print(f"Generated {len(manifest)} transformed example images")

    else:
        raise ValueError("Either --manifest or --input_dir must be provided.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = load_model(
        checkpoint_path,
        device,
    )

    results = []

    print("\nRunning predictions...\n")

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Predicting",
    ):

        image_path = Path(
            row["transformed_path"]
        )

        try:
            prediction, ai_probability = predict_image(
                model,
                image_path,
                device,
            )

            label_name = (
                "AI generated"
                if prediction == 1
                else "real"
            )

            results.append({
                "image_path": str(image_path),
                "pred": label_name,
                "ai_probability": ai_probability,
                "transform": row["transform"],
            })

        except Exception as error:
            print(
                f"\nSkipping image:\n"
                f"{image_path}\n"
                f"Error: {error}"
            )

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.suffix.lower() == ".json":
        results_df.to_json(
            output_path,
            orient="records",
            indent=2,
        )
    else:
        results_df.to_csv(
            output_path,
            index=False,
        )

    print("\nFinished!")
    print(f"Predictions saved to:\n{output_path}")
    print(f"Successfully predicted {len(results_df)} images")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run a trained AI image detector "
            "on robustness test images or a folder of example images."
        )
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Path to robustness manifest CSV. "
            "Use this for saved transformed images."
        ),
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Directory of example images to augment on the fly and score."
        ),
    )

    parser.add_argument(
        "--transforms",
        nargs="*",
        default=DEFAULT_TRANSFORMS,
        help=(
            "Transforms to apply when --input_dir is used. "
            "Options: " + ", ".join(DEFAULT_TRANSFORMS)
        ),
    )

    parser.add_argument(
        "--temp_dir",
        default="data/robustness_examples",
        help=(
            "Directory used to store transformed example images before scoring."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to trained .pth checkpoint"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/predictions.csv"
        ),
        help=(
            "Path to output predictions CSV"
        ),
    )

    args = parser.parse_args()

    if not args.manifest and not args.input_dir:
        parser.error("Either --manifest or --input_dir is required.")

    main(args)