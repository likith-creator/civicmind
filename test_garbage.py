from ultralytics import YOLO

model = YOLO("models/civicmind_garbage.pt")

results = model.predict(
    source="garbage dataset/test/images",
    conf=0.25,
    save=True
)

print("Garbage detection test completed.")
print("Detection results saved in the runs folder.")