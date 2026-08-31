from torchvision import transforms


# Used for the original / clean model
clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Used when training the augmented model
augmented_transform = transforms.Compose([
    # Make images large enough before cropping
    transforms.Resize((256, 256)),

    # Random crop and resize back to 224x224
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0),
    ),

    # Random left-right flip
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # Occasionally apply blur
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0),
        )
    ], p=0.3),

    # Small colour changes
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),

    # Small rotation
    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Used for validation, testing, and inference.
# No random augmentation here.
evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])