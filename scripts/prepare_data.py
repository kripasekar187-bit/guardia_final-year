"""
scripts/prepare_data.py
-----------------------
Turn raw per-class .npy files into sequences ready for the TCN model.

HOW TO USE:
  python scripts/prepare_data.py

  Reads:  data/normal.npy, data/fall.npy, data/inactive.npy
  Writes: data/X.npy  (shape: samples × TIME_STEPS × 132)
          data/y.npy  (shape: samples,  labels: 0=normal 1=fall 2=inactive)

WHY SEQUENCES?
  A single frame can't tell you if someone is falling — it could just be
  them bending down. A TCN (Temporal Convolutional Network) looks at a
  sliding window of TIME_STEPS consecutive frames to capture motion over time.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from guardia.config import DATA_DIR

TIME_STEPS = 30   # ~1 second at 30fps


def create_sequences(data, name):
    if len(data) <= TIME_STEPS:
        print(f"\nERROR: '{name}' only has {len(data)} frames — need more than {TIME_STEPS}.")
        print(f"Run:  python scripts/collect_data.py  and record '{name}' for a few seconds.\n")
        sys.exit(1)

    sequences = []
    for i in range(len(data) - TIME_STEPS):
        sequences.append(data[i : i + TIME_STEPS])
    return np.array(sequences)


# ── Load raw data ──────────────────────────────────────────────────────────
try:
    normal   = np.load(os.path.join(DATA_DIR, "normal.npy"))
    fall     = np.load(os.path.join(DATA_DIR, "fall.npy"))
    inactive = np.load(os.path.join(DATA_DIR, "inactive.npy"))
except FileNotFoundError as e:
    print(f"\nERROR: Missing dataset file — {e}")
    print("Run collect_data.py for each of: normal, fall, inactive\n")
    sys.exit(1)

# ── Build sequences ────────────────────────────────────────────────────────
X_normal   = create_sequences(normal,   "normal")
X_fall     = create_sequences(fall,     "fall")
X_inactive = create_sequences(inactive, "inactive")

X = np.concatenate([X_normal, X_fall, X_inactive])
y = np.concatenate([
    np.zeros(len(X_normal)),          # label 0 = normal
    np.ones(len(X_fall)),             # label 1 = fall
    np.full(len(X_inactive), 2),      # label 2 = inactive
])

# ── Save ───────────────────────────────────────────────────────────────────
np.save(os.path.join(DATA_DIR, "X.npy"), X)
np.save(os.path.join(DATA_DIR, "y.npy"), y)
print(f"Dataset ready — X: {X.shape}  y: {y.shape}")
