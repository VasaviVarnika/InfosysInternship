import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("videos/video.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully.")

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Frame width: {frame_width}, Frame height: {frame_height}, FPS: {fps}")

output_video = cv2.VideoWriter(
    "output/tracking_obj_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height)
)

# Initialize variables
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached or error reading frame.")
        break

    count += 1
    print(f"Processing frame {count}")

    # Object detection
    (class_ids, scores, boxes) = od.detect(frame)
    print(f"Detected {len(boxes)} objects in frame {count}")

    # Center points of current frame
    center_points_cur_frame = []

    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        print(f"Object center: ({cx}, {cy})")

    # Debug tracking logic here
    # ...

    # Write the frame to the output video
    output_video.write(frame)

cap.release()
output_video.release()
cv2.destroyAllWindows()