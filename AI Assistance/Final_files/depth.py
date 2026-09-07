from ultralytics import YOLO
import cv2
import torch
import numpy as np
from transformers import pipeline
from tts import speak

model = YOLO(r"C:\Users\User\OneDrive\Desktop\AI Assistance\Final_files\best.pt")

# Depth Anything V2 (small = fastest variant, still heavy on CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if device == "cuda" else -1
)

# Depth Anything gives RELATIVE inverse depth (0-1, higher = closer),
# not metric meters. These thresholds need calibration on YOUR camera —
# see notes below.
CLOSE_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35


def get_object_depth(depth_map, x1, y1, x2, y2):
    """Median depth value inside a bounding box. Robust to edge noise."""
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))


def object_detection(camera):
    spoken_objects = set()
    frame_count = 0
    depth_map = None
    DEPTH_EVERY_N_FRAMES = 3  # run depth less often than detection

    while True:
        ret, frame = camera.read()
        if not ret:
            return False

        frame_count += 1
        height, width, _ = frame.shape

        # --- Run depth estimation (throttled) ---
        if frame_count % DEPTH_EVERY_N_FRAMES == 0 or depth_map is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_result = depth_estimator(rgb_frame)
            depth_map = np.array(depth_result["depth"])
            # normalize to 0-1 so thresholds stay consistent frame to frame
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

        # --- Run YOLO ---
        results = model.predict(frame, verbose=False)
        current_objects = set()

        for r in results:
            for box in r.boxes:
                x, y, w, h = box.xywh[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                current_objects.add(class_name)

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                if x < width / 3:
                    direction = "LEFT"
                elif x < 2 * width / 3:
                    direction = "MIDDLE"
                else:
                    direction = "RIGHT"

                depth_value = get_object_depth(depth_map, x1, y1, x2, y2)

                if depth_value > CLOSE_THRESHOLD:
                    distance_status = "CLOSE"
                elif depth_value > MEDIUM_THRESHOLD:
                    distance_status = "MEDIUM"
                else:
                    distance_status = "FAR"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} | {direction} | {distance_status}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if distance_status != "FAR" and class_name not in spoken_objects:
                    speak(f"{class_name} on the {direction}, {distance_status}")
                    spoken_objects.add(class_name)

        spoken_objects = spoken_objects.intersection(current_objects)

        cv2.imshow("Avanti Object Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyWindow("Avanti Object Detection")
            return True