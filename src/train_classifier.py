
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights
)

from torch.utils.data import DataLoader
from pathlib import Path


# -------------------------------
# 1. Configuration
# -------------------------------

DEVICE = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 2

print("Training device:", DEVICE)


# -------------------------------
# 2. Load pretrained weights
# -------------------------------

weights = MobileNet_V3_Large_Weights.DEFAULT

model = mobilenet_v3_large(weights=weights)


# -------------------------------
# 3. Image transformations
# -------------------------------

normalization = transforms.Normalize(
    mean=weights.transforms().mean,
    std=weights.transforms().std
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalization
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalization
])


# -------------------------------
# 4. Load datasets
# -------------------------------

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=eval_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -------------------------------
# 5. Create DataLoaders
# -------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -------------------------------
# 6. Freeze pretrained layers
# -------------------------------

for param in model.features.parameters():
    param.requires_grad = False


# Replace final classification layer
input_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# -------------------------------
# 7. Loss and optimizer
# -------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# -------------------------------
# 8. Training loop
# -------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update classifier weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # -------------------------------
    # 9. Validation
    # -------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = val_loss / val_total

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


# -------------------------------
# 10. Save trained model
# -------------------------------

Path("models").mkdir(exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_dataset.classes,
    "model_name": "mobilenet_v3_large"
}, "models/cat_dog_classifier.pth")

print("\nModel training completed!")
print("Model saved successfully!")
