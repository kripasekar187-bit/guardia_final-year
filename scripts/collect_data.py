"""
Record pose keypoints from the webcam and save as a .npy file.

Usage:
  python scripts/collect_data.py
  Enter label: fall / normal / inactive
  Move in front of the camera, then press Q.
  Saves to: data/{label}.npy
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import mediapipe as mp

from guardia.config import CAMERA_INDEX, DATA_DIR

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose()

label = input("Enter label (fall / normal / inactive): ").strip()
os.makedirs(DATA_DIR, exist_ok=True)

cap  = cv2.VideoCapture(CAMERA_INDEX)
data = []

print(f"Recording '{label}' — press Q when done.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        data.append(keypoints)

    cv2.imshow(f"Collecting '{label}' — Q to stop", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

data     = np.array(data)
out_path = os.path.join(DATA_DIR, f"{label}.npy")

if os.path.exists(out_path):
    existing = np.load(out_path)
    data     = np.concatenate([existing, data])
    print(f"Appended to existing data")

np.save(out_path, data)
print(f"Saved {data.shape} → {out_path}")
