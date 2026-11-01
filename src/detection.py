
from ultralytics import YOLO
from pathlib import Path
import time


class ObjectDetector:

    def __init__(self):

        # CPU device
        self.device = "cpu"

        # Load pretrained YOLO26 nano model
        self.model = YOLO("yolo26n.pt")

        print("YOLO26n loaded successfully!")


    def predict(self, image_path):

        # Check image existence
        if not Path(image_path).is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Run object detection
        start_time = time.perf_counter()

        results = self.model.predict(
            source=image_path,
            device=self.device,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        inference_time = time.perf_counter() - start_time

        result = results[0]

        print("\nOBJECT DETECTION RESULTS")
        print("-" * 40)

        # Extract bounding boxes
        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())

                confidence = float(box.conf.item())

                class_name = self.model.names[class_id]

                coordinates = box.xyxy[0].tolist()

                print(f"\nObject: {class_name}")
                print(f"Confidence: {confidence * 100:.2f}%")
                print(f"Bounding Box: {coordinates}")

        else:
            print("No objects detected.")

        # Save annotated image
        output_path = Path("outputs/detection/result.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result.save(filename=str(output_path))

        print("\nSaved output:", output_path)
        print(f"Inference Time: {inference_time:.4f} seconds")


if __name__ == "__main__":

    detector = ObjectDetector()

    detector.predict("dataset/sample.jpg")
