import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)

model = YOLO("yolo11n.pt")
model("hero.png", verbose=False)
print("Model is running on:", model.device)
import torch
print(torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))