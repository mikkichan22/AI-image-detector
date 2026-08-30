from pathlib import Path
import argparse
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from src.transforms import (
    augmented_transform,
    evaluation_transform,
)

class ManifestDataset(Dataset):
    """
    Dataset that reads image paths and labels from data/splits.csv.
    """

    def __init__(self, dataframe, project_root, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])

        if not image_path.exists():
            image_path = self.project_root / image_path

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return image, label

# for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# compares correct labels with model's preductions
def calculate_metrics(labels, predictions):
    return {
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
    }

# Trainnig/ evaluating one epoch
def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    labels = []
    predictions = []
    ai_probabilities = []

    for images, batch_labels in tqdm(loader, leave=False):
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, batch_labels)

            if training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)
        batch_predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        labels.extend(batch_labels.cpu().numpy())
        predictions.extend(batch_predictions.cpu().numpy())
        ai_probabilities.extend(
            probabilities[:, 1].detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(labels, predictions)
    metrics["loss"] = average_loss
    metrics["labels"] = labels
    metrics["predictions"] = predictions
    metrics["ai_probabilities"] = ai_probabilities

    return metrics


def build_model():
    # loads ResNet-18 model pretrained on ImageNet
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )
    # the number of inputs to ResNet’s final layer.
    number_of_features = model.fc.in_features

    # Replaces the original final layer with one that predicts two classes: 0 = real, 1 = AI/ Fake
    model.fc = nn.Linear(
        number_of_features,
        2,
    )

    return model


def main(args):
    set_seed(args.seed)

    project_root = Path(args.project_root)
    splits_path = project_root / args.splits_file
    output_dir = project_root / args.output_dir
    checkpoint_dir = project_root / args.checkpoint_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # run on GPU if present, if not CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    print("Reading split file:", splits_path)

    splits = pd.read_csv(splits_path)

    train_rows = splits[splits["split"] == "train"]
    validation_rows = splits[splits["split"] == "validation"]
    test_rows = splits[splits["split"] == "test"]

    train_dataset = ManifestDataset(
        train_rows,
        project_root,
        augmented_transform,
    )

    validation_dataset = ManifestDataset(
        validation_rows,
        project_root,
        evaluation_transform,
    )

    test_dataset = ManifestDataset(
        test_rows,
        project_root,
        evaluation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    # Initially train only the new classification layer.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=args.learning_rate,
    )

    best_validation_f1 = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Fine-tune the full model after the frozen phase.
        if epoch == args.freeze_epochs:
            print("Unfreezing full ResNet-18")

            for parameter in model.parameters():
                parameter.requires_grad = True

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.fine_tune_learning_rate,
            )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
        }

        history.append(record)

        print(
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Validation loss: {validation_metrics['loss']:.4f}, "
            f"Validation F1: {validation_metrics['f1']:.4f}"
        )

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

            checkpoint_path = (
                checkpoint_dir / "resnet18_clean_best.pth"
            )

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "validation_f1": best_validation_f1,
                    "class_mapping": {
                        "0": "real",
                        "1": "AI/fake",
                    },
                },
                checkpoint_path,
            )

            print("Saved:", checkpoint_path)

    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    # Load the best validation checkpoint.
    model.load_state_dict(best_state)

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=["real", "AI/fake"],
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=[0, 1],
    )

    final_metrics = {
        "model": "resnet18",
        "training_type": "clean",
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    metrics_path = output_dir / "test_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(final_metrics, file, indent=2)

    print("\nFinal test results:")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1:        {test_metrics['f1']:.4f}")

    print("\nSaved:")
    print(history_path)
    print(metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project_root",
        default=".",
    )

    parser.add_argument(
        "--splits_file",
        default="data/splits.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="results/resnet18_clean",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--fine_tune_learning_rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())