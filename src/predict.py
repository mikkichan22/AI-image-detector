from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

try:
    from src.transforms import evaluation_transform
except ModuleNotFoundError:
    from transforms import evaluation_transform


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_model():
    """
    Build the same ResNet-18 architecture used during training.
    """
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        2,
    )

    return model


def load_checkpoint(checkpoint_path, device):
    """
    Load the trained checkpoint.
    """
    model = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Your checkpoint contains model_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        print("Checkpoint information:")
        print(
            "  Epoch:",
            checkpoint.get("epoch", "unknown")
        )
        print(
            "  Validation F1:",
            checkpoint.get("validation_f1", "unknown")
        )
        print(
            "  Class mapping:",
            checkpoint.get("class_mapping", "unknown")
        )
    else:
        # Fallback in case you later use a checkpoint
        # containing only the model state dictionary.
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def find_images(input_dir):
    """
    Find all supported image files inside the input directory,
    including subfolders.
    """
    input_dir = Path(input_dir)

    images = []

    for path in input_dir.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(path)

    return sorted(images)


def predict_image(model, image_path, device):
    """
    Predict the probability that an image belongs
    to class 1 (AI-generated/manipulated).
    """
    image = Image.open(image_path).convert("RGB")

    image_tensor = evaluation_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        ai_probability = probabilities[0, 1].item()

        prediction = outputs.argmax(
            dim=1
        ).item()

    return prediction, ai_probability


def main(args):
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Loading checkpoint:", checkpoint_path)

    model = load_checkpoint(
        checkpoint_path,
        device,
    )

    image_paths = find_images(input_dir)

    print(
        f"Found {len(image_paths)} supported images"
    )

    results = []

    for image_path in image_paths:
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
                "prediction": label_name,
                "ai_probability": round(ai_probability, 6),
            })

            print(
                f"{image_path} | "
                f"{label_name} | "
                f"AI probability={ai_probability:.4f}"
            )

        except Exception as error:
            print(
                f"Skipping {image_path}: {error}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved {len(results)} predictions "
        f"to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run AI image detection on a directory "
            "of images."
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing images",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint",
    )

    args = parser.parse_args()

    main(args)