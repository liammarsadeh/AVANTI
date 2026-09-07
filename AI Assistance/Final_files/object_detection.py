# from ultralytics import YOLO
# import cv2
# from tts import speak

# model = YOLO(r"C:\Users\User\OneDrive\Desktop\AI Assistance\Final_files\best.pt")



# def object_detection(camera):
#     spoken_objects = set()

#     while True:
#         ret, frame = camera.read()

#         if not ret:
#             return False

#         results = model.predict(frame, verbose=False,conf=0.4)

#         current_objects = set()

#         height, width, _ = frame.shape

#         for r in results:
#             for box in r.boxes:

#                 x, y, w, h = box.xywh[0]

#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]

#                 current_objects.add(class_name)

#                 area = w * h

#                 if x < width / 3:
#                     direction = "LEFT"
#                 elif x < 2 * width / 3:
#                     direction = "MIDDLE"
#                 else:
#                     direction = "RIGHT"

#                 if area < 20000:
#                     distance_status = "FAR"
#                 elif area < 60000:
#                     distance_status = "MEDIUM"
#                 else:
#                     distance_status = "CLOSE"

#                 x1 = int(x - w / 2)
#                 y1 = int(y - h / 2)
#                 x2 = int(x + w / 2)
#                 y2 = int(y + h / 2)

#                 cv2.rectangle(
#                     frame,
#                     (x1, y1),
#                     (x2, y2),
#                     (0, 255, 0),
#                     2
#                 )

#                 cv2.putText(
#                     frame,
#                     f"{class_name} | {direction} | {distance_status}",
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2
#                 )

#                 # Speak only if this object hasn't been announced
#                 if distance_status != "FAR" and class_name not in spoken_objects:
#                     speak(f"{class_name} on the {direction}, {distance_status}")
#                     spoken_objects.add(class_name)

#         # Objects that disappeared can be announced again later
#         spoken_objects = spoken_objects.intersection(current_objects)

#         cv2.imshow("Avanti Object Detection", frame)

#         if cv2.waitKey(1) == ord("q"):
#             cv2.destroyWindow("Avanti Object Detection")
#             return True







from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from tts import speak


model = YOLO(
    r"C:\Users\User\OneDrive\Desktop\AI Assistance\Final_files\best.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if device == "cuda" else -1
)


# Relative depth (0-1, higher = closer)

CLOSE_THRESHOLD = 0.8
MEDIUM_THRESHOLD = 0.5


def get_object_depth(depth_map, x1, y1, x2, y2):

    h, w = depth_map.shape

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # median depth inside the bounding box
    return float(
        np.median(depth_map[y1:y2, x1:x2])
    )


def object_detection(camera):

    # If an object has been detected,
    # don't announce it again until it disappears.
    spoken_objects = set()

    while True:

        # Read camera frame
        ret, frame = camera.read()

        if not ret:
            return False

        height, width, _ = frame.shape


        # DEPTH ESTIMATION PRE-PROCESSING STEPS!!!

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb_frame)

        # Pass image to depth model
        depth_result = depth_estimator(pil_image)

        # Convert depth image to NumPy
        depth_map = np.array(
            depth_result["depth"]
        )

        # Normalize depth to 0-1
        # 0 = Far
        # 1 = Close
        depth_map = (
            depth_map - depth_map.min()
        ) / (
            depth_map.max()
            - depth_map.min()
            + 1e-6
        )

        # Resize depth map to camera resolution since camera in main loop has (1920x1080)
        depth_map = cv2.resize(
            depth_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )


        # YOLO DETECTION 
        results = model.predict(
            frame,
            verbose=False,
            conf=0.3
        )

        current_objects = set()


        for r in results:

            for box in r.boxes:

                # Bounding box
                x, y, w, h = box.xywh[0]

                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)


                # Object class
                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                current_objects.add(class_name)


                # BOUNDING BOX COORDINATES SO WE CAN DRAW RECTANGLE (BOUNDING BOX)
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)

                x2 = int(x + w / 2)
                y2 = int(y + h / 2)


                # DIRECTION OF THE OBJECT BY DIVIDING THE SCREEN
                if x < width / 3:

                    direction = "LEFT"

                elif x < 2 * width / 3:

                    direction = "MIDDLE"

                else:

                    direction = "RIGHT"


                # GET DEPTH VALUE APPROX (0-1)
                depth_value = get_object_depth(
                    depth_map,
                    x1,
                    y1,
                    x2,
                    y2
                )


                # DISTANCE STATUS
                if depth_value > CLOSE_THRESHOLD:

                    distance_status = "CLOSE"

                elif depth_value > MEDIUM_THRESHOLD:

                    distance_status = "MEDIUM"

                else:

                    distance_status = "FAR"


                
                # DRAW BOUNDING BOX
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # DISPLAY INFORMATION TO LABEL BOUNDING BOX
                cv2.putText(
                    frame,
                    f"{class_name} | "
                    f"{direction} | "
                    f"{distance_status}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                
                # SPEAK OBJECT IF THEY ARE NOT FAR (CLOSE, MEDIUM)
                if (
                    distance_status != "FAR"
                    and class_name not in spoken_objects
                ):

                    speak(
                        f"{class_name} on the "
                        f"{direction}, "
                        f"{distance_status}"
                    )

                    spoken_objects.add(class_name)


        
        # RESET SPOKEN OBJECTS

        # Objects that disappeared can
        # be announced again later.

        spoken_objects = (
            spoken_objects.intersection(current_objects)
        )

        # Display Video
        cv2.imshow(
            "Avanti Object Detection",
            frame
        )


        
        # EXIT
        if cv2.waitKey(1) == ord("q"):

            cv2.destroyWindow(
                "Avanti Object Detection"
            )

            return True