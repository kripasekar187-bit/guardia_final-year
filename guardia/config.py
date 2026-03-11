"""
guardia/config.py
-----------------
Central configuration for GUARDIA.

WHY THIS FILE EXISTS:
  Magic numbers scattered across code are hard to tune and easy to forget.
  Keeping every threshold and setting here means you change ONE place
  and every module that imports it automatically gets the update.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR   = os.path.join(ROOT_DIR, "logs")

ALERTS_LOG  = os.path.join(LOGS_DIR, "alerts.txt")
FALL_IMAGE  = os.path.join(LOGS_DIR, "fall_event.jpg")
MODEL_PATH  = os.path.join(MODELS_DIR, "tcn_model.h5")

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0

# ── MediaPipe Pose ─────────────────────────────────────────────────────────
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5

# ── Fall Detection ─────────────────────────────────────────────────────────
# How far the head landmark must drop between frames to count as a fall.
# Landmark coordinates are normalised 0-1 relative to frame height.
FALL_THRESHOLD = 0.12

# ── Activity / Inactivity ──────────────────────────────────────────────────
# Minimum normalised movement to consider the person "active".
MOVEMENT_THRESHOLD = 0.02

# Seconds of no movement before an inactivity alert fires.
INACTIVITY_THRESHOLD = 15

# ── Alert Timing ───────────────────────────────────────────────────────────
# Seconds the system waits for the user to press Y before escalating.
ALERT_RESPONSE_TIMEOUT = 7

# ── Night Mode ─────────────────────────────────────────────────────────────
# Mean pixel brightness (0-255) below which night mode activates.
NIGHT_BRIGHTNESS_THRESHOLD = 60
NIGHT_ALPHA = 1.8   # contrast multiplier
NIGHT_BETA  = 40    # brightness additive

# ── Text-to-Speech ─────────────────────────────────────────────────────────
TTS_RATE = 150   # words per minute
