
from pathlib import Path
import sys

from ultralytics import YOLO


# Project paths
MODEL_PATH = Path("models/yolo26n.pt")
OUTPUT_DIR = Path("outputs/detection")

DEVICE = "cpu"
CONFIDENCE = 0.25
IMAGE_SIZE = 640


def detect_objects(image_path):

    image_path = Path(image_path)

    # Check model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH.resolve()}"
        )

    # Check image
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load YOLO26
    model = YOLO(str(MODEL_PATH))

    print("YOLO26 model loaded successfully!")

    # Perform detection
    results = model.predict(
        source=str(image_path),
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    for result in results:

        print("\n===== DETECTION RESULTS =====")

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")

        else:
            for box in result.boxes:

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                label = result.names[class_id]

                print("Object:", label)
                print("Confidence:", round(confidence * 100, 2), "%")
                print(
                    "Bounding Box:",
                    [round(x1, 2), round(y1, 2),
                     round(x2, 2), round(y2, 2)]
                )
                print("-----------------------")

        # Save annotated image
        output_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"

        result.save(filename=str(output_path))

        print("\nSaved output:", output_path)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python src/detection.py <image_path>"
        )
        sys.exit(1)

    detect_objects(sys.argv[1])
