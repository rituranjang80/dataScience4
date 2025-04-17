from ultralytics import YOLO
import cv2
import time
import json

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.
video_path = "1.mp4"
output_video = "1output_video.mp4"
log_file = "timestamps.json"

# Objects to detect (COCO class names)
TARGET_OBJECTS = ["person", "book", "dog", "car"]  # Add more as needed
# COCO class IDs for these objects (man=person, mango=not in COCO, mobile=cell phone)
CLASS_NAMES = model.names
CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items() if name in TARGET_OBJECTS}

# Dictionary to store timestamps
detection_timestamps = {obj: [] for obj in TARGET_OBJECTS}

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get current timestamp (in seconds)
    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Detect objects
    results = model(frame, classes=list(CLASS_IDS.values()), conf=0.5)

    # Log detections
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = CLASS_NAMES[class_id]
        if class_name in TARGET_OBJECTS:
            detection_timestamps[class_name].append(round(frame_time, 2))

    # Draw results on frame
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

# Save timestamps to JSON
with open(log_file, "w") as f:
    json.dump(detection_timestamps, f, indent=4)

cap.release()
out.release()
print(f"Saved annotated video to {output_video}")
print(f"Detection timestamps saved to {log_file}")