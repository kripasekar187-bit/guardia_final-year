"""
scripts/collect_data.py
-----------------------
Record pose keypoints from the webcam and save them as a .npy file.

HOW TO USE:
  python scripts/collect_data.py

  When prompted, enter one of:  fall / normal / inactive
  Move in front of the camera for a few seconds, then press Q.
  The recorded data is saved to:  data/{label}.npy

HOW IT WORKS:
  MediaPipe detects 33 body landmarks per frame.
  Each landmark has x, y, z, and visibility — giving 33×4 = 132 values per frame.
  We stack these into a 2D array (frames × 132) and save it.
"""

import sys
import os

# ── Make sure the project root is on the Python path ──────────────────────
# This lets us run the script from any directory.
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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(frame_rgb)

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

data = np.array(data)
out_path = os.path.join(DATA_DIR, f"{label}.npy")
np.save(out_path, data)
print(f"Saved {data.shape} → {out_path}")
