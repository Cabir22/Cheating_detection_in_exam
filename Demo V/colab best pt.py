import zipfile

with zipfile.ZipFile("/content/Yolo Phone V.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")

from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="/content/dataset/Yolo Phone V/data.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    device=0
)