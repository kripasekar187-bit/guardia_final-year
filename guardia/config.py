import os

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR   = os.path.join(ROOT_DIR, "logs")

ALERTS_LOG = os.path.join(LOGS_DIR, "alerts.txt")
FALL_IMAGE = os.path.join(LOGS_DIR, "fall_event.jpg")
MODEL_PATH  = os.path.join(MODELS_DIR, "tcn_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

CAMERA_INDEX = 0

POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5

FALL_THRESHOLD       = 0.12
MOVEMENT_THRESHOLD   = 0.02
INACTIVITY_THRESHOLD = 15      # seconds

ALERT_RESPONSE_TIMEOUT = 7     # seconds before escalating

NIGHT_BRIGHTNESS_THRESHOLD = 60
NIGHT_ALPHA = 1.8
NIGHT_BETA  = 40

TTS_RATE = 150
