# This script performs:
# i) Dataset download from Roboflow
# ii) YOLOv8 training
# iii) Upload trained model to S3 

from roboflow import Roboflow
from ultralytics import YOLO
import boto3

# Download dataset from Roboflow
rf = Roboflow(api_key="ROBOFLOW_API_KEY")
project = rf.workspace("perception-models").project("tata_ace_exterior")
version = project.version(2)
dataset = version.download("yolov8")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=dataset.location + "/data.yaml",
    epochs=2,
    imgsz=320,
    batch=2,
    workers=0
)

# Upload trained model to S3
s3 = boto3.client('s3')
s3.upload_file(
    "runs/detect/train/weights/best.pt",
    "your-s3-bucket-name",
    "models/best.pt"
)

print("Training completed and model uploaded to S3")

# test pipeline
