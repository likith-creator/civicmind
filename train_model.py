from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs",
    name="civicmind_pothole"
)

print("Training completed successfully!")