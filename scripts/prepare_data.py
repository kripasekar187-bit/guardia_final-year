"""
Build training sequences from raw .npy files.

Usage:
  python scripts/prepare_data.py

Reads:  data/normal.npy, data/fall.npy, data/inactive.npy
Writes: data/X.npy  (samples × 30 × 132)
        data/y.npy  (0=normal, 1=fall, 2=inactive)
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from guardia.config import DATA_DIR

TIME_STEPS = 30
STEP = 10  # non-overlapping enough to prevent train/val leakage


def create_sequences(data, name):
    if len(data) <= TIME_STEPS:
        print(f"ERROR: '{name}' only has {len(data)} frames — need more than {TIME_STEPS}.")
        sys.exit(1)
    sequences = []
    for i in range(0, len(data) - TIME_STEPS, STEP):
        sequences.append(data[i : i + TIME_STEPS])
    return np.array(sequences)


try:
    normal   = np.load(os.path.join(DATA_DIR, "normal.npy"))
    fall     = np.load(os.path.join(DATA_DIR, "fall.npy"))
    inactive = np.load(os.path.join(DATA_DIR, "inactive.npy"))
except FileNotFoundError as e:
    print(f"ERROR: Missing file — {e}")
    sys.exit(1)

X_normal   = create_sequences(normal,   "normal")
X_fall     = create_sequences(fall,     "fall")
X_inactive = create_sequences(inactive, "inactive")

X = np.concatenate([X_normal, X_fall, X_inactive])
y = np.concatenate([
    np.zeros(len(X_normal)),
    np.ones(len(X_fall)),
    np.full(len(X_inactive), 2),
])

np.save(os.path.join(DATA_DIR, "X.npy"), X)
np.save(os.path.join(DATA_DIR, "y.npy"), y)
print(f"Dataset ready — X: {X.shape}  y: {y.shape}")
