
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

import sys


DEVICE = "cpu"

# Load saved checkpoint
checkpoint = torch.load(
    "models/cat_dog_classifier.pth",
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]

# Recreate model architecture
model = mobilenet_v3_large(weights=None)

input_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    input_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = preprocess(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.inference_mode():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    label = classes[predicted.item()]

    print("\nPrediction:", label)

    print(
        "Confidence:",
        round(confidence.item() * 100, 2),
        "%"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict_custom.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(image_path)
