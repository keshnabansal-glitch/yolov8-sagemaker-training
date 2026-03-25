# This script performs:
# i) Dataset download from Roboflow
# ii) YOLOv8 training
# iii) Upload trained model to S3 

from roboflow import Roboflow
from ultralytics import YOLO
import boto3
import os
import sys

print("Starting YOLOv8 training pipeline...")

# Get API key from environment (passed from GitHub)
try:
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("perception-models").project("tata_ace_exterior")
    version = project.version(2)
    print("Downloading dataset from Roboflow...")
    dataset = version.download("yolov8")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Train the model
print("Starting training...")
try:
    model.train(
        data=dataset.location + "/data.yaml",
        epochs=2,
        imgsz=320,
        batch=2,
        workers=0
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)

# Upload trained model to S3
try:
    s3 = boto3.client('s3')
    # Get correct path dynamically
    best_model_path = str(model.trainer.save_dir / "weights/best.pt")
    print(f"Model saved at: {best_model_path}")
    
    # Check if file exists
    import os.path
    if os.path.exists(best_model_path):
        print(f"Uploading {best_model_path} to S3...")
        s3.upload_file(
            best_model_path,
            "yolov8-trained-model",
            "models/best.pt"
        )
        print("Model uploaded to S3 successfully!")
    else:
        print(f"Warning: Model file not found at {best_model_path}")
        # List what files do exist
        import glob
        pt_files = glob.glob("runs/**/*.pt", recursive=True)
        print(f"Found PT files: {pt_files}")
except Exception as e:
    print(f"Warning: S3 upload failed (artifact will still be available in GitHub): {e}")

print("Training pipeline completed!")
