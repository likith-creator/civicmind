from ultralytics import YOLO
from pathlib import Path

# Load our trained pothole model
model = YOLO("models/civicmind_pothole.pt")

# Get the first image from the test dataset
test_dir = Path("dataset/test/images")
image_path = next(
    p for p in test_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

print("Testing image:", image_path)

# Run detection
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True,
    project="runs",
    name="pothole_test",
    exist_ok=True,
    verbose=False
)

# Display detections
for result in results:
    if len(result.boxes) == 0:
        print("❌ No pothole detected")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                f"✅ Detected: {model.names[class_id]} "
                f"| Confidence: {confidence * 100:.2f}%"
            )

print("Test completed!")