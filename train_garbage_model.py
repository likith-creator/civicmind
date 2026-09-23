from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train garbage detection model
model.train(
    data="garbage dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    project="runs",
    name="civicmind_garbage"
)

print("Garbage training completed successfully!")