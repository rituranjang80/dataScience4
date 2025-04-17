from ultralytics import YOLO
import cv2
import numpy as np
import json
import time

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # or custom model for guns
video_path = "2.mp4"
output_video = "2output_video.mp4"
log_file = "detections.json"

# Target objects and colors (HSV ranges)
TARGETS = {
    "person_white_shirt": {"class": "person", "colors": {"shirt": ([0, 0, 200], [180, 50, 255])}},  # White shirt (BGR->HSV)
    "person_black_pants": {"class": "person", "colors": {"pants": ([0, 0, 0], [180, 255, 50])}},    # Black pants
    "red_mobile": {"class": "cell phone", "colors": {"mobile": ([0, 100, 100], [10, 255, 255])}},   # Red mobile
    "gun": {"class": "gun", "colors": None}  # Gun (no color filter)
}

# HSV color ranges for filtering
def get_color_mask(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower), np.array(upper))

# Main detection
def detect_objects():
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    detections = {key: [] for key in TARGETS.keys()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        results = model(frame, classes=[0, 67, 76], conf=0.5)  # person, cell phone, gun

        for obj_name, config in TARGETS.items():
            class_name = config["class"]
            for box in results[0].boxes:
                if model.names[int(box.cls)] == class_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]

                    # Color filtering
                    if config["colors"]:
                        for part, (lower, upper) in config["colors"].items():
                            mask = get_color_mask(roi, lower, upper)
                            if cv2.countNonZero(mask) > 50:  # Minimum pixels to confirm color
                                detections[obj_name].append(round(current_time, 2))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{obj_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Gun in hand (check proximity to person)
                    if obj_name == "gun":
                        for person_box in results[0].boxes:
                            if model.names[int(person_box.cls)] == "person":
                                px1, py1, px2, py2 = map(int, person_box.xyxy[0])
                                gun_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                hand_zone = (px1, py1 - 50, px2, py1 + 50)  # Near hands
                                if (hand_zone[0] < gun_center[0] < hand_zone[2] and 
                                    hand_zone[1] < gun_center[1] < hand_zone[3]):
                                    detections["gun"].append(round(current_time, 2))
                                    cv2.putText(frame, "GUN IN HAND", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    with open(log_file, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"Detections saved to {log_file}")

if __name__ == "__main__":
    detect_objects()