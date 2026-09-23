from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

results = model.predict(
    source="test_images/test.jpg",
    conf=0.25,
    save=True
)

print("Detection completed!")