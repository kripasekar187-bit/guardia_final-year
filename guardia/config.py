"""
guardia/config.py - Central configuration for thresholds and API keys.
"""

import os

# --- Paths ---

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR   = os.path.join(ROOT_DIR, "logs")

ALERTS_LOG = os.path.join(LOGS_DIR, "alerts.txt")
FALL_IMAGE = os.path.join(LOGS_DIR, "fall_event.jpg")
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
# Upgrade path: swap to yolov8s-pose.pt or yolov8m-pose.pt for
# better keypoint accuracy on slower hardware (yolov8n is fastest).


# --- Camera ---

CAMERA_INDEX = 0          # 0 = default webcam.  Change to 1, 2 … for USB/IP cams.


# --- Detection Thresholds ---

# Fall velocity threshold
FALLING_VELOCITY_THRESHOLD = 0.12

# Primary signal 2: Aspect ratio (bounding box W/H)
ASPECT_RATIO_THRESHOLD = 0.84

# Spine angle threshold (degrees from vertical)
SPINE_ANGLE_THRESHOLD_DEG = 45.0

# Signals required to trigger alert
FALL_SIGNAL_COUNT_REQUIRED = 2

# --- Velocity smoothing ----------------------------------------------
# EMA alpha for centroid velocity.
# Higher = faster response, more noise.
# Lower  = smoother, more latency.
# 0.35 ≈ 3-frame effective window at 30 fps.
VELOCITY_EMA_ALPHA = 0.35

# Min keypoint confidence
MIN_KP_CONF = 0.30

# --- Recovery timer --------------------------------------------------
# Seconds the person must remain in a fallen posture before the alert
# is locked in as a TRUE emergency.
# 3.5 s gives enough time to catch themselves after a stumble without
# false-alarming.  For very frail or bedridden users, lower to 2.0 s.
RECOVERY_TIMEOUT_SECONDS = 3.5

# Max horizontal inactivity (seconds)
INACTIVITY_THRESHOLD = 8

# --- Activity motion gate --------------------------------------------
# Minimum normalised displacement (torso_y delta / box_height) that
# counts as "real movement" for resetting the inactivity timer.
# Below this = micro-jitter noise; do not reset the timer.
# 0.03 ≈ 3% of body height per frame.
ACTIVITY_MOTION_THRESHOLD = 0.03

# --- Stale track timeout ---------------------------------------------
# Seconds after last detection before a person's state is purged.
# Prevents unbounded memory growth when people leave the frame.
STALE_TRACK_TIMEOUT = 5.0

# --- Stability Fixes -------------------------------------------------
# Number of consecutive frames a hand must be raised to cancel an alert.
# Prevents single-frame keypoint noise from clearing emergencies.
GESTURE_CONFIRM_FRAMES = 10 

# Seconds a person must stay UP-RIGHT before a fallen alert is cleared.
# Prevents alert oscillation if keypoints flicker while lying down.
RECOVERY_LATCH_SECONDS = 2.0


# --- Alert & Response ---

# Seconds after the first "Are you okay?" TTS before escalating to
# the full alarm + log.  Give enough time for the person to respond.
ALERT_RESPONSE_TIMEOUT = 10

# --- SMS Alert (Twilio) ----------------------------------------------
# Set to True once you have your Twilio credentials
ENABLE_SMS = True

# Twilio Credentials
TWILIO_ACCOUNT_SID = "AC6121cafe191c4679edba98f34c79c237"
TWILIO_AUTH_TOKEN  = "4f13aea5563ccb6eae163ed172c5ed84"

# Twilio Phone Number
TWILIO_FROM_NUMBER = "+18103775029"

# Recipient Number
EMERGENCY_CONTACT_NUMBER = "+919778176499"

# --- Rich Media (ImgBB) ----------------------------------------------
# API Key from https://api.imgbb.com/
IMGBB_API_KEY = "4f832985f377cdfc85761ae927997a96"

# --- WhatsApp Alert --------------------------------------------------
# Enable this to send alerts via WhatsApp (requires Twilio Sandbox)
ENABLE_WHATSAPP = True
WHATSAPP_TO_NUMBER = "whatsapp:+919778176499"
WHATSAPP_FROM_NUMBER = "whatsapp:+14155238886"  # Default Twilio Sandbox number

# Number of alarm beeps to play during escalation.
ALARM_BEEP_COUNT = 5

# Beep frequency (Hz) and duration (ms) for winsound.Beep.
ALARM_BEEP_FREQ = 1200
ALARM_BEEP_MS   = 800

# Text-to-speech speaking rate (words per minute).
TTS_RATE = 150

# Spoken alert messages
TTS_ALERT_MESSAGE = "Are you okay?"
TTS_ESCALATION_MESSAGE = "Emergency detected. Please help."
TTS_OK_MESSAGE = "Okay, alert cancelled."

# --- Night Mode (brightness enhancement) ---
# Mean pixel brightness below this value triggers enhancement.
NIGHT_BRIGHTNESS_THRESHOLD = 60

# Contrast multiplier and brightness offset
NIGHT_ALPHA = 1.8
NIGHT_BETA  = 40