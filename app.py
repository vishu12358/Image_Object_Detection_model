# Streamlit application


import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from ultralytics import YOLO


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Image Analysis System",
    page_icon="🤖",
    layout="wide"
)

DEVICE = "cpu"

CLASSIFIER_PATH = Path("models/cat_dog_classifier.pth")
YOLO_PATH = Path("models/yolo26n.pt")


# ---------------- LOAD CLASSIFICATION MODEL ----------------

@st.cache_resource
def load_classifier():

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = mobilenet_v3_large(weights=None)

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, classes


# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_detector():

    model = YOLO(str(YOLO_PATH))

    return model


# ---------------- IMAGE PREPROCESSING ----------------

preprocess = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ---------------- CLASSIFICATION ----------------

def classify_image(image):

    model, classes = load_classifier()

    image_tensor = preprocess(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.inference_mode():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    label = classes[predicted.item()]

    return label, confidence.item() * 100


# ---------------- OBJECT DETECTION ----------------

def detect_objects(image, confidence_threshold):

    model = load_detector()

    results = model.predict(
        source=image,
        device=DEVICE,
        imgsz=640,
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    # YOLO plot returns a BGR image
    annotated_image = result.plot()

    # Convert BGR to RGB for Streamlit
    annotated_image = annotated_image[:, :, ::-1]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())

            confidence = float(box.conf.item())

            label = result.names[class_id]

            coordinates = [
                round(value, 2)
                for value in box.xyxy[0].tolist()
            ]

            detections.append({
                "Object": label,
                "Confidence": round(confidence * 100, 2),
                "Bounding Box": coordinates
            })

    return annotated_image, detections


# ---------------- MAIN APPLICATION ----------------

st.title("🤖 AI Image Analysis System")

st.write(
    "Upload an image to classify cats and dogs "
    "or detect multiple objects using AI."
)

st.divider()


# Sidebar
st.sidebar.title("Settings")

mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    [
        "Image Classification",
        "Object Detection",
        "Both"
    ]
)

confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.info(
    "Classification: MobileNetV3-Large\n\n"
    "Object Detection: YOLO26n\n\n"
    "Device: CPU"
)


# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Original Image",
        width=500
    )

    if st.button("Analyze Image", type="primary"):

        try:

            # Classification mode
            if mode in ["Image Classification", "Both"]:

                with st.spinner("Running classification..."):

                    label, confidence = classify_image(image)

                st.subheader("Classification Results")

                st.success(f"Prediction: {label}")

                st.metric(
                    "Classification Confidence",
                    f"{confidence:.2f}%"
                )


            # Detection mode
            if mode in ["Object Detection", "Both"]:

                with st.spinner("Detecting objects..."):

                    annotated_image, detections = detect_objects(
                        image,
                        confidence_threshold
                    )

                st.subheader("Object Detection Results")

                st.image(
                    annotated_image,
                    caption="YOLO26 Detection Output",
                    width=700
                )

                if detections:

                    st.write("Detected Objects")

                    st.dataframe(
                        detections,
                        use_container_width=True
                    )

                else:

                    st.warning("No objects detected.")

        except FileNotFoundError as error:

            st.error(str(error))

        except Exception as error:

            st.error(f"An error occurred: {error}")

else:

    st.info("Please upload an image to begin analysis.")

st.divider()

st.caption(
    "AI Image Analysis System | "
    "MobileNetV3 + YOLO26 | CPU Inference"
)
