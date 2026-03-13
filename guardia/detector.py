import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from collections import deque

import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model

from guardia.config import (
    POSE_DETECTION_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    INACTIVITY_THRESHOLD,
    MODEL_PATH,
    SCALER_PATH,
)

_mp_pose = mp.solutions.pose
_mp_draw  = mp.solutions.drawing_utils

# labels from training
_LABELS = ["normal", "fall", "inactive"]


class GuardiaDetector:

    def __init__(self):
        self.pose = _mp_pose.Pose(
            min_detection_confidence=POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
        )

        try:
            self._model  = load_model(MODEL_PATH, compile=False)
            self._scaler = joblib.load(SCALER_PATH)
            print("Model loaded")
        except Exception as e:
            print(f"Model not found, falling back to heuristic: {e}")
            self._model  = None
            self._scaler = None

        self._frame_buffer    = deque(maxlen=30)
        self._prediction      = "normal"
        self._confirm_counter = 0
        self._CONFIRM_NEEDED  = 5     # consecutive frames before alert
        self._CONFIDENCE_MIN  = 0.65  # minimum model confidence
        self._reset_state()

    @property
    def emergency(self):
        return self._prediction in ("fall", "inactive")

    @property
    def prediction(self):
        return self._prediction

    def process(self, frame_rgb):
        return self.pose.process(frame_rgb)

    def update(self, landmarks):
        frame_features = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
        self._frame_buffer.append(frame_features)

        if len(self._frame_buffer) == 30:
            if self._model is not None:
                self._predict_ml()
            else:
                self._predict_heuristic(landmarks)

    def _predict_ml(self):
        X = np.array(self._frame_buffer)
        X = self._scaler.transform(X)
        X = X[np.newaxis, ...]
        probs = self._model.predict(X, verbose=0)[0]
        confidence = np.max(probs)
        label      = _LABELS[np.argmax(probs)]

        if label != "normal" and confidence >= self._CONFIDENCE_MIN:
            self._confirm_counter += 1
        else:
            self._confirm_counter = 0
            self._prediction = "normal"

        if self._confirm_counter >= self._CONFIRM_NEEDED:
            self._prediction = label

    def _predict_heuristic(self, landmarks):
        # fallback if model unavailable
        pose_array = np.array([[lm.x, lm.y] for lm in landmarks])
        movement   = np.linalg.norm(pose_array - pose_array.mean(axis=0))
        if movement < 0.02:
            if time.time() - self._last_movement_time > INACTIVITY_THRESHOLD:
                self._prediction = "inactive"
                return
        else:
            self._last_movement_time = time.time()

        head_y = landmarks[0].y
        if self._prev_head_y is not None and head_y - self._prev_head_y > 0.12:
            self._prediction = "fall"
        else:
            self._prediction = "normal"
        self._prev_head_y = head_y

    def draw_landmarks(self, frame, pose_landmarks):
        _mp_draw.draw_landmarks(frame, pose_landmarks, _mp_pose.POSE_CONNECTIONS)

    def reset(self):
        self._prediction      = "normal"
        self._confirm_counter = 0
        self._frame_buffer.clear()
        self._reset_state()

    def _reset_state(self):
        self._prev_head_y        = None
        self._last_movement_time = time.time()
