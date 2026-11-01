
import sys
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import sklearn
import ultralytics

print("=" * 45)
print("IMAGE DETECTION PROJECT")
print("=" * 45)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Ultralytics:", ultralytics.__version__)

print("\nPyTorch CPU Available:", torch.backends.cpu.get_cpu_capability())
print("CUDA Available:", torch.cuda.is_available())

print("\nEnvironment setup completed!")
