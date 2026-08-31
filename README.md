# AI Image Detector

## Overview

This project is a lightweight AI-generated image detector prototype built with PyTorch and a ResNet-18 backbone. It classifies whether an input image is likely to be real or AI-generated/manipulated, while also allowing evaluation after common real-world transformations such as JPEG compression, blur, cropping, resizing, noise, and colour changes.

The repository includes:

- a prediction script that accepts an input directory of images and outputs JSON predictions;
- a robustness evaluation script that applies controlled transformations before scoring images;
- two pretrained checkpoints:
  - `clean_best`: trained on clean CIFAKE images;
  - `sid_best`: trained using a SID-priority subset together with CIFAKE data and training-time augmentation.

## Problem Context

Generative AI models can now produce highly realistic images at scale, creating risks related to misinformation, impersonation, fraud, and trust in digital content. A practical detector should perform well on clean images and remain useful after common post-processing operations such as compression, resizing, blur, cropping, and colour adjustment.

This prototype focuses on a hackathon-scale solution: fast to run, easy to evaluate, and suitable for demonstrating image-level AI detection with public datasets and realistic image transformations.

## What this repo contains

- `src/predict.py` — scores a directory of image files and saves JSON output with one result per image;
- `src/predict_robustness.py` — applies robustness transformations to input images and evaluates them with the same model;
- `src/transforms.py` — training and evaluation preprocessing pipelines;
- `src/create_robustness_sets.py` — creates transformed robustness samples for test-time analysis;
- `src/create_splits.py` — creates reproducible CIFAKE data splits;
- `src/train_resnet.py` — trains and evaluates the ResNet-18 classifier;
- `src/evaluate.py` — evaluates predictions and calculates classification metrics;
- `checkpoints/resnet18_clean_best.pth` — clean CIFAKE checkpoint;
- `checkpoints/resnet18_sid_best.pth` — SID-priority augmented checkpoint;
- `examples/` — sample images for quick testing.

The final repository should include the `checkpoints/` and `examples/` directories. They are not present in the current source zip, so add them before publishing.

## Data sources and labels

The project uses public image datasets relevant to AI image detection:

- **CIFAKE:** real and AI-generated synthetic images;
- **SID_Set:** real, fully synthetic, and tampered images;
- **WildFake validation subset:** external demonstration-only evaluation data supplied for the hackathon.

The binary label mapping is:

- `0 = real`;
- `1 = AI-generated or AI-manipulated`.

For the current SID experiment, a balanced subset of 10,000 real and 10,000 fully synthetic images was used. SID's tampered label was excluded from this experiment, so the current model should not be described as fully evaluated on tampered images.

The `clean_best` checkpoint was trained on clean CIFAKE images. The `sid_best` checkpoint was trained with SID-priority data combined with sampled CIFAKE data. Its training pipeline included random resized crops, horizontal flips, colour jitter, Gaussian blur, JPEG compression, and Gaussian noise.

## Reported results

The following results were recorded for the SID-priority augmented model:

| Evaluation set | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Combined internal test set, 23,377 images | 0.9445 | 0.9263 | 0.9678 | 0.9466 |
| SID held-out test subset, 3,000 images | 0.9930 | 0.9881 | 0.9980 | 0.9930 |
| CIFAKE test portion, 20,378 images | 0.9384 | 0.9193 | 0.9637 | 0.9410 |
| WildFake demonstration subset, 600 images | 0.8000 | 0.9688 | 0.6200 | 0.7561 |

The WildFake result shows a cross-dataset generalisation limitation: precision was high, but recall was lower. A final clean-versus-transformed robustness table should be generated from the final checkpoint before submission.

## Setup

1. Clone the repository.

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux, activate it with:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running inference on a directory of images

From the repository root:

```powershell
python src/predict.py --input_dir examples --output results/example_predictions.json --checkpoint checkpoints/resnet18_clean_best.pth
```

To use the SID-priority checkpoint instead:

```powershell
python src/predict.py --input_dir examples --output results/example_predictions_sid.json --checkpoint checkpoints/resnet18_sid_best.pth
```

This scans supported image files under `examples`, runs the model, and writes one JSON record per image.

Example output:

```json
[
  {
    "image_path": "examples\\ai_example.jpg",
    "prediction": "AI generated",
    "ai_probability": 0.982341
  },
  {
    "image_path": "examples\\real_example.jpg",
    "prediction": "real",
    "ai_probability": 0.143702
  }
]
```

The current script outputs `prediction` and `ai_probability`. The hackathon brief refers to a required field named `pred`; if the evaluator checks the schema strictly, update `src/predict.py` to emit `pred` as well, then rerun and inspect the JSON before submission.

## Robustness evaluation on transformed images

The robustness script supports:

- JPEG compression (`jpeg_70`, `jpeg_30`);
- Gaussian blur (`blur_1.0`);
- resizing and upscaling (`resize_0.5`);
- Gaussian noise (`noise_0.05`);
- colour adjustment (`colour_jitter`);
- center cropping and resizing (`crop_80`).

Evaluate a saved labelled robustness manifest:

```powershell
python src/predict_robustness.py --manifest data/robustness_manifest.csv --checkpoint checkpoints/resnet18_sid_best.pth --output results/robustness_predictions.csv
```

Or transform example images during inference:

```powershell
python src/predict_robustness.py --input_dir examples --checkpoint checkpoints/resnet18_sid_best.pth --output results/example_robustness.json --transforms clean jpeg_70 jpeg_30 blur_1.0 resize_0.5 noise_0.05 colour_jitter crop_80
```

The output contains:

- `image_path`;
- `prediction`;
- `ai_probability`;
- `transform`.

For the submission, calculate accuracy, precision, recall, and F1 separately for each transform and include a compact clean-versus-transformed table in the written project description.

## Checkpoints

The project ships with two pretrained ResNet-18 checkpoints:

- `checkpoints/resnet18_clean_best.pth` — trained on clean CIFAKE data and intended as the clean-image baseline;
- `checkpoints/resnet18_sid_best.pth` — trained using the SID-priority data mixture and augmentation pipeline, and intended as the stronger final demo candidate.

Both checkpoints use the same two-class ResNet-18 architecture and can be swapped in the inference commands using `--checkpoint`.

## Notes and limitations

- This repository is intended as a practical prototype and evaluation tool rather than a full production moderation pipeline.
- The model's softmax output is an estimated score, not a calibrated probability or proof of provenance.
- Public datasets may contain source-specific shortcuts, so performance may not transfer to unseen generators or real-world social-media images.
- SID tampered images were not included in the current SID-priority training experiment.
- The external WildFake result demonstrates reduced recall on a different data distribution.
- The frequency-branch functionality was removed from the final scope because of time constraints and is not part of the main inference path.
- The detector analyses still images only and does not cover video or audio.

## Quick command summary

```powershell
# Clean CIFAKE checkpoint
python src/predict.py --input_dir examples --output results/example_predictions_clean.json --checkpoint checkpoints/resnet18_clean_best.pth

# SID-priority augmented checkpoint
python src/predict.py --input_dir examples --output results/example_predictions_sid.json --checkpoint checkpoints/resnet18_sid_best.pth

# Robust transformed evaluation
python src/predict_robustness.py --input_dir examples --checkpoint checkpoints/resnet18_sid_best.pth --output results/example_robustness.json --transforms clean jpeg_70 jpeg_30 blur_1.0 resize_0.5 noise_0.05 colour_jitter crop_80
```

## Team contributions

Replace this section with the actual team member names and contributions before publishing.

## Dataset citations and licence

Add the official dataset links, citations, repository licence, and any required dataset licence notices before making the repository public. Do not commit raw datasets, credentials, or private local paths.
