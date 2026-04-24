import cv2
from pathlib import Path
from ultralytics import YOLO

weights = Path("c:/Users/junio/OneDrive/Documents/semester 7/SKRIPSI/coding/pengujian/YOLOv8/outputs/runs/yolov8s/train_20260418-000303/weights/best.pt")
model = YOLO(str(weights))

# Process the first few frames of the video
cap = cv2.VideoCapture("c:/Users/junio/OneDrive/Documents/semester 7/SKRIPSI/coding/pengujian/YOLOv8/data/k.mp4")

for i in range(5):
    ret, frame = cap.read()
    if not ret: break
    
    results = model.track(source=frame, conf=0.15, persist=True, tracker="c:/Users/junio/OneDrive/Documents/semester 7/SKRIPSI/coding/pengujian/YOLOv8/scripts/custom_tracker.yaml", verbose=False)
    
    r = results[0]
    print(f"--- Frame {i} ---")
    if r.boxes is not None:
        print(f"Number of boxes: {len(r.boxes)}")
        if r.boxes.id is not None:
            print(f"Number of IDs: {len(r.boxes.id)}")
            print(f"IDs: {r.boxes.id.tolist()}")
        else:
            print("r.boxes.id is None")
    else:
        print("No boxes")

cap.release()
