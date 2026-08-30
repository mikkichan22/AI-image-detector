import io
import random

import numpy as np
from PIL import Image
from torchvision import transforms


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class RandomJPEGCompression:
    def __init__(self, probability=0.3, quality_range=(30, 90)):
        self.probability = probability
        self.quality_range = quality_range

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        quality = random.randint(
            self.quality_range[0],
            self.quality_range[1],
        )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        return Image.open(buffer).convert("RGB")


class RandomGaussianNoise:
    def __init__(self, probability=0.2, sigma_range=(0.01, 0.05)):
        self.probability = probability
        self.sigma_range = sigma_range

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        sigma = random.uniform(
            self.sigma_range[0],
            self.sigma_range[1],
        )

        array = np.asarray(image).astype(np.float32) / 255.0
        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=array.shape,
        )

        noisy = np.clip(array + noise, 0.0, 1.0)

        return Image.fromarray(
            (noisy * 255).astype(np.uint8)
        )


clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])


augmented_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        size=224,
        scale=(0.8, 1.0),
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            )
        ],
        p=0.5,
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.5, 2.0),
            )
        ],
        p=0.3,
    ),

    transforms.RandomApply(
        [
            RandomJPEGCompression(
                probability=1.0,
                quality_range=(30, 90),
            )
        ],
        p=0.3,
    ),

    transforms.RandomApply(
        [
            RandomGaussianNoise(
                probability=1.0,
                sigma_range=(0.01, 0.05),
            )
        ],
        p=0.2,
    ),

    transforms.ToTensor(),
    NORMALIZE,
])


evaluation_transform = clean_transform