"""
guardia/detector.py
-------------------
GuardiaDetector — wraps MediaPipe Pose and owns all detection state.

WHY A CLASS?
  The original main.py used module-level variables (prev_head_y, fall_detected,
  last_movement_time) that got tangled with display and alert code.
  A class bundles the state it needs with the methods that use it, making
  each piece independently readable and testable.
"""

import time
import numpy as np
import mediapipe as mp

from guardia.config import (
    POSE_DETECTION_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    FALL_THRESHOLD,
    MOVEMENT_THRESHOLD,
    INACTIVITY_THRESHOLD,
)

# MediaPipe helpers (module-level — these are stateless utilities)
_mp_pose = mp.solutions.pose
_mp_draw  = mp.solutions.drawing_utils


class GuardiaDetector:
    """
    Handles pose estimation, fall detection, and inactivity monitoring.

    Usage:
        detector = GuardiaDetector()
        results  = detector.process(frame_rgb)

        if results.pose_landmarks:
            detector.check_fall(results.pose_landmarks.landmark)
            inactivity = detector.check_inactivity(results.pose_landmarks.landmark)
    """

    def __init__(self):
        self.pose = _mp_pose.Pose(
            min_detection_confidence=POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
        )
        self._reset_state()

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def fall_detected(self):
        return self._fall_detected

    @property
    def inactivity_seconds(self):
        return time.time() - self._last_movement_time

    @property
    def emergency(self):
        """True when a fall OR prolonged inactivity is detected."""
        return self._fall_detected or self.inactivity_seconds > INACTIVITY_THRESHOLD

    # ── Core methods ───────────────────────────────────────────────────────

    def process(self, frame_rgb):
        """Run MediaPipe on an RGB frame. Returns a pose results object."""
        return self.pose.process(frame_rgb)

    def check_fall(self, landmarks):
        """
        Compare the head landmark's Y position between consecutive frames.
        A sudden downward drop (increase in Y) flags a fall.

        Landmark Y is normalised 0-1 (0 = top of frame, 1 = bottom),
        so a large positive delta means the head moved toward the floor fast.
        """
        head_y = landmarks[0].y
        if self._prev_head_y is not None:
            drop = head_y - self._prev_head_y
            if drop > FALL_THRESHOLD:
                self._fall_detected = True
        self._prev_head_y = head_y

    def check_inactivity(self, landmarks):
        """
        Compute overall pose movement by measuring how spread the landmarks
        are around their centroid. If spread exceeds the threshold the person
        is considered active and the inactivity timer resets.

        Returns the number of seconds since last detected movement.
        """
        pose_array = np.array([[lm.x, lm.y] for lm in landmarks])
        movement   = np.linalg.norm(pose_array - pose_array.mean(axis=0))
        if movement > MOVEMENT_THRESHOLD:
            self._last_movement_time = time.time()
        return self.inactivity_seconds

    def draw_landmarks(self, frame, pose_landmarks):
        """Draw the skeleton overlay onto the frame (modifies in place)."""
        _mp_draw.draw_landmarks(frame, pose_landmarks, _mp_pose.POSE_CONNECTIONS)

    def reset(self):
        """Call this after an alert is resolved to clear detection state."""
        self._reset_state()

    # ── Private ────────────────────────────────────────────────────────────

    def _reset_state(self):
        self._prev_head_y        = None
        self._fall_detected      = False
        self._last_movement_time = time.time()
