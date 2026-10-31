# Made By Nitish Bhatt ML-2
# Class Roll Number 55
# Graphic Era Demmed to Be University

!pip install -q ultralytics opencv-python-headless

import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
from google.colab import files
from IPython.display import display, Video

print("Loading YOLO model...")
model = YOLO("yolo11n-pose.pt")

print("Upload your dance video:")
uploaded = files.upload()
input_video = next(iter(uploaded.keys()))

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or width <= 0 or height <= 0:
    raise ValueError("Invalid video file")

print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total Frames:", total_frames)

output_video = "skeleton_dance.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("Could not create output video")

skeleton_color = (255, 255, 0)
joint_color = (255, 255, 255)

confidence = 0.35
keypoint_confidence = 0.25
glow_intensity = 0.8

alpha = 0.65
previous_keypoints = None
selected_id = None

connections = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def draw_skeleton(keypoints, confidences):

    skeleton = np.zeros((height, width, 3), dtype=np.uint8)
    glow = np.zeros_like(skeleton)

    points = []

    for i, point in enumerate(keypoints):
        x, y = map(int, point)

        if confidences[i] >= keypoint_confidence:
            points.append((x, y))
        else:
            points.append(None)

    for start, end in connections:

        p1 = points[start]
        p2 = points[end]

        if p1 is not None and p2 is not None:

            cv2.line(
                glow, p1, p2,
                skeleton_color, 12, cv2.LINE_AA
            )

            cv2.line(
                skeleton, p1, p2,
                skeleton_color, 3, cv2.LINE_AA
            )

    for point in points:

        if point is not None:

            cv2.circle(
                glow, point, 10,
                joint_color, -1, cv2.LINE_AA
            )

            cv2.circle(
                skeleton, point, 5,
                joint_color, -1, cv2.LINE_AA
            )

    blurred = cv2.GaussianBlur(glow, (0, 0), 8)

    result = cv2.addWeighted(
        skeleton, 1.0,
        blurred, glow_intensity,
        0
    )

    return result

print("Starting video processing...")

frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=confidence,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]
    output_frame = np.zeros_like(frame)

    if (
        result.keypoints is not None
        and len(result.keypoints.data) > 0
    ):

        keypoints_all = result.keypoints.data.cpu().numpy()
        boxes = result.boxes

        chosen_index = None

        if boxes.id is not None:

            ids = boxes.id.cpu().numpy().astype(int)

            if selected_id is None:
                selected_id = ids[0]

            if selected_id in ids:
                chosen_index = int(
                    np.where(ids == selected_id)[0][0]
                )

        if chosen_index is None:

            areas = (
                boxes.xyxy[:, 2] - boxes.xyxy[:, 0]
            ) * (
                boxes.xyxy[:, 3] - boxes.xyxy[:, 1]
            )

            chosen_index = int(areas.argmax())

            if boxes.id is not None:
                selected_id = int(ids[chosen_index])

            previous_keypoints = None

        keypoints = keypoints_all[chosen_index].copy()

        coordinates = keypoints[:, :2]
        point_confidence = keypoints[:, 2]

        if previous_keypoints is not None:

            valid = (
                (point_confidence >= keypoint_confidence) &
                (previous_keypoints[:, 2] >= keypoint_confidence)
            )

            coordinates[valid] = (
                alpha * coordinates[valid] +
                (1 - alpha) * previous_keypoints[valid, :2]
            )

        keypoints[:, :2] = coordinates
        previous_keypoints = keypoints.copy()

        output_frame = draw_skeleton(
            coordinates,
            point_confidence
        )

    else:
        previous_keypoints = None

    out.write(output_frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames")

cap.release()
out.release()

print("Video processing completed")

subprocess.run([
    "ffmpeg", "-y",
    "-i", output_video,
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "skeleton_dance_final.mp4"
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Final skeleton video:")

display(Video(
    "skeleton_dance_final.mp4",
    embed=True,
    width=700
))

files.download("skeleton_dance_final.mp4")

print("Done!")
