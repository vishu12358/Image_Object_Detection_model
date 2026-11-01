
import torch
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights
)
from PIL import Image
import time


class ImageClassifier:

    def __init__(self):

        self.device = torch.device("cpu")

        # Load pretrained ImageNet weights
        self.weights = MobileNet_V3_Large_Weights.DEFAULT

        self.model = mobilenet_v3_large(
            weights=self.weights
        )

        self.model.to(self.device)
        self.model.eval()

        # Preprocessing pipeline supplied with weights
        self.preprocess = self.weights.transforms()

        # ImageNet class labels
        self.categories = self.weights.meta["categories"]

        print("MobileNetV3 loaded successfully!")


    def predict(self, image_path):

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        input_tensor = self.preprocess(image)

        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Inference
        start_time = time.perf_counter()

        with torch.inference_mode():

            output = self.model(input_batch)

            probabilities = F.softmax(output, dim=1)

        inference_time = time.perf_counter() - start_time

        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(
            probabilities[0], 5
        )

        print("\nIMAGE CLASSIFICATION RESULTS")
        print("-" * 40)

        for probability, index in zip(
            top5_prob, top5_indices
        ):

            label = self.categories[index.item()]
            confidence = probability.item() * 100

            print(f"{label}: {confidence:.2f}%")

        print(f"\nInference Time: {inference_time:.4f} seconds")


if __name__ == "__main__":

    classifier = ImageClassifier()

    classifier.predict("dataset/sample.jpg")
