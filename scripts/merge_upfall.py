"""
Merges the UP-Fall dataset (MediaPipe 33 joints, x/y/z) into our
existing fall.npy / normal.npy files, then rewrites them.

UP-Fall label convention:
  0 = no fall (ADL frame)  → our 'normal' class
  1 = fall impact frame     → our 'fall' class

Feature mismatch fix:
  UP-Fall has 99 features (33 joints × 3).
  Our model expects 132 (33 joints × 4, with visibility).
  We pad visibility = 1.0 for all UP-Fall joints.

Run from project root:
  python scripts/merge_upfall.py
"""

import os, sys, re, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPFALL_DIR = os.path.join(ROOT, "data", "upfall")
DATA_DIR   = os.path.join(ROOT, "data")

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    """Load a CSV, normalise the label column name."""
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        upper = c.strip().upper()
        if upper in ("LABEL", "LLABEL"):
            rename[c] = "LABEL"
        elif c == "0":
            rename[c] = "LABEL"
    if rename:
        df = df.rename(columns=rename)
    return df


def add_visibility(arr_99):
    """
    Convert (N, 99) → (N, 132) by inserting visibility=1.0 after every 3 coords.
    MediaPipe layout per joint: x, y, z, visibility
    """
    n = arr_99.shape[0]
    out = np.ones((n, 132), dtype=np.float32)
    for j in range(33):
        out[:, j*4 + 0] = arr_99[:, j*3 + 0]   # x
        out[:, j*4 + 1] = arr_99[:, j*3 + 1]   # y
        out[:, j*4 + 2] = arr_99[:, j*3 + 2]   # z
        out[:, j*4 + 3] = 1.0                   # visibility (assumed visible)
    return out

# ── load UP-Fall CSVs ─────────────────────────────────────────────────────────

files = glob.glob(os.path.join(UPFALL_DIR, "*.csv"))
print(f"Found {len(files)} UP-Fall CSV files")

upfall_fall   = []   # LABEL == 1
upfall_normal = []   # LABEL == 0
skipped = 0

for path in files:
    df = load_csv(path)
    if "LABEL" not in df.columns:
        skipped += 1
        continue

    feat_cols = [c for c in df.columns if c != "LABEL"]
    if len(feat_cols) != 99:
        skipped += 1
        continue

    features = df[feat_cols].values.astype(np.float32)
    labels   = df["LABEL"].values

    fall_frames   = features[labels == 1]
    normal_frames = features[labels == 0]

    if len(fall_frames):
        upfall_fall.append(add_visibility(fall_frames))
    if len(normal_frames):
        upfall_normal.append(add_visibility(normal_frames))

print(f"Skipped {skipped} malformed files")

upfall_fall   = np.vstack(upfall_fall)   if upfall_fall   else np.empty((0,132))
upfall_normal = np.vstack(upfall_normal) if upfall_normal else np.empty((0,132))

print(f"UP-Fall → fall frames:   {len(upfall_fall)}")
print(f"UP-Fall → normal frames: {len(upfall_normal)}")

# ── load existing data ────────────────────────────────────────────────────────

existing_fall     = np.load(os.path.join(DATA_DIR, "fall.npy"))
existing_normal   = np.load(os.path.join(DATA_DIR, "normal.npy"))
existing_inactive = np.load(os.path.join(DATA_DIR, "inactive.npy"))

print(f"\nExisting fall:     {existing_fall.shape}")
print(f"Existing normal:   {existing_normal.shape}")
print(f"Existing inactive: {existing_inactive.shape}")

# ── merge ─────────────────────────────────────────────────────────────────────

merged_fall   = np.vstack([existing_fall,   upfall_fall])
merged_normal = np.vstack([existing_normal, upfall_normal])

print(f"\nMerged fall:     {merged_fall.shape}")
print(f"Merged normal:   {merged_normal.shape}")
print(f"Inactive:        {existing_inactive.shape}  (unchanged)")

# ── save ──────────────────────────────────────────────────────────────────────

np.save(os.path.join(DATA_DIR, "fall.npy"),     merged_fall)
np.save(os.path.join(DATA_DIR, "normal.npy"),   merged_normal)
# inactive.npy stays as-is

print("\nSaved fall.npy and normal.npy")
print("\nNext step: python scripts/prepare_data.py")
