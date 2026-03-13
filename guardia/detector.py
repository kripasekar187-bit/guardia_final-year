"""
guardia/detector.py - Fall detection logic using YOLOv8-pose.
"""

import time
import math
import numpy as np
from collections import deque
from ultralytics import YOLO

from guardia.config import (
    MODEL_PATH,
    FALLING_VELOCITY_THRESHOLD,
    ASPECT_RATIO_THRESHOLD,
    INACTIVITY_THRESHOLD,
    GESTURE_CONFIRM_FRAMES,
    RECOVERY_LATCH_SECONDS,
)

# --- Constants ---
MIN_KP_CONF = 0.30
VELOCITY_EMA_ALPHA = 0.35
SPINE_ANGLE_THRESHOLD_DEG = 45.0
TORSO_VELOCITY_THRESHOLD = FALLING_VELOCITY_THRESHOLD
RECOVERY_TIMEOUT_SECONDS = 3.5
STALE_TRACK_TIMEOUT = 5.0
ACTIVITY_MOTION_THRESHOLD = 0.03


class GuardiaDetector:
    """
    Multi-person fall detector using YOLOv8-pose with BoT-SORT tracking.
    """

    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"[GUARDIA] YOLOv8-pose loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[GUARDIA] Failed to load model: {e}")
            self.model = None

        # Per-person state keyed by BoT-SORT track ID
        self.person_states: dict = {}

        # Global output prediction
        self._global_prediction: str = "normal"

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def emergency(self) -> bool:
        return self._global_prediction in ("fall", "inactive")

    @property
    def prediction(self) -> str:
        return self._global_prediction

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray):
        """Run YOLOv8 pose-track on an RGB frame.  Returns Results list."""
        if self.model is None:
            return None
        return self.model.track(frame_rgb, persist=True, verbose=False)

    def update(self, results) -> None:
        """Update person states from the latest Results object."""
        if not results or len(results) == 0:
            return

        result = results[0]

        if len(result.boxes) == 0 or result.boxes.id is None:
            return

        current_time = time.time()

        for i, box in enumerate(result.boxes):
            track_id = int(box.id.item())

            if track_id not in self.person_states:
                self.person_states[track_id] = _create_empty_state()

            state = self.person_states[track_id]
            state['last_seen'] = current_time

            # Only run heuristics if we have keypoints for this detection
            if result.keypoints and i < len(result.keypoints.xy):
                kp_xy   = result.keypoints.xy[i]    # shape (17, 2)
                kp_conf = (
                    result.keypoints.conf[i]         # shape (17,)
                    if result.keypoints.conf is not None
                    else None
                )
                if len(kp_xy) > 0:
                    self._analyse_person(
                        track_id, state, box, kp_xy, kp_conf, current_time
                    )

        self._cleanup_stale_tracks(current_time)
        self._update_global_prediction()

    # ------------------------------------------------------------------
    # Per-person analysis
    # ------------------------------------------------------------------

    def _analyse_person(self, track_id, state, box,
                        kp_xy, kp_conf, current_time):
        """
        Run all heuristics for a single tracked person and advance
        their state machine.
        """
        # Bounding box geometry
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_w = float(x2 - x1)
        box_h = float(y2 - y1)
        if box_h <= 0:
            return

        def kp(idx):
            """Return (x, y) if keypoint confidence is sufficient, else None."""
            if kp_conf is not None:
                conf = float(kp_conf[idx].cpu().numpy())
                if conf < MIN_KP_CONF:
                    return None
            x = float(kp_xy[idx][0].cpu().numpy())
            y = float(kp_xy[idx][1].cpu().numpy())
            # Coordinates of (0, 0) mean the keypoint was not detected
            if x < 1.0 and y < 1.0:
                return None
            return (x, y)

        # Aspect Ratio
        aspect_ratio   = box_w / box_h
        is_horizontal  = aspect_ratio > ASPECT_RATIO_THRESHOLD

        # Spine Angle
        spine_angle_deg = None
        is_spine_tilted = False

        l_shoulder = kp(5);  r_shoulder = kp(6)
        l_hip      = kp(11); r_hip      = kp(12)

        if l_shoulder and r_shoulder and l_hip and r_hip:
            sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
            sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            hp_x = (l_hip[0]      + r_hip[0])      / 2.0
            hp_y = (l_hip[1]      + r_hip[1])      / 2.0

            # Vector from shoulder to hip (downward in image = +y)
            dx = hp_x - sh_x
            dy = hp_y - sh_y
            spine_len = math.hypot(dx, dy)

            if spine_len > 5.0:   # guard against co-located ghosts
                # Angle from vertical (dy-axis)
                # atan2(|dx|, |dy|): 0 = vertical, 90 = horizontal
                spine_angle_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
                is_spine_tilted = spine_angle_deg > SPINE_ANGLE_THRESHOLD_DEG

                # Override is_horizontal using real spine geometry.
                # If spine is strongly vertical, box-shape is NOT a fall.
                spine_norm = (hp_y - sh_y) / box_h
                if spine_norm > 0.30:           # long vertical spine segment
                    is_horizontal = False

            state['torso_sh'] = (sh_x, sh_y)
            state['torso_hp'] = (hp_x, hp_y)

        # Centroid Velocity (EMA)
        is_falling_fast = False
        torso_y = None

        if state.get('torso_hp') and state.get('torso_sh'):
            hp = state['torso_hp'];  sh = state['torso_sh']
            torso_y = (hp[1] + sh[1]) / 2.0
        else:
            # Fallback: use bounding box centroid
            torso_y = (y1 + y2) / 2.0

        if state['prev_torso_y'] is not None and state['prev_time'] is not None:
            dt = current_time - state['prev_time']
            if dt > 0:
                # Normalise displacement by box height → scale-invariant
                raw_delta = (torso_y - state['prev_torso_y']) / box_h
                raw_velocity = raw_delta / dt

                # EMA smooth
                prev_ema = state['velocity_ema']
                if prev_ema is None:
                    ema_velocity = raw_velocity
                else:
                    ema_velocity = (VELOCITY_EMA_ALPHA * raw_velocity
                                    + (1.0 - VELOCITY_EMA_ALPHA) * prev_ema)
                state['velocity_ema'] = ema_velocity

                is_falling_fast = ema_velocity > TORSO_VELOCITY_THRESHOLD

                # Activity tracking: only count as "active" if the
                # person is NOT already horizontal (prevents micro-jitter
                # from resetting the inactivity timer after a fall).
                if abs(raw_delta) > ACTIVITY_MOTION_THRESHOLD and not is_horizontal:
                    state['last_active_time'] = current_time

        state['prev_torso_y'] = torso_y
        state['prev_time']    = current_time

        # Crouch/Bending Check
        is_crouching = False

        l_ankle = kp(15); r_ankle = kp(16)

        if l_hip and r_hip and (l_ankle or r_ankle):
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
            ankle_ys = [a[1] for a in [l_ankle, r_ankle] if a]
            ankle_y  = sum(ankle_ys) / len(ankle_ys)

            # Standing: ankle_y >> hip_y (large positive gap)
            # Crouching/on-ground: ankle_y ≈ hip_y (small gap)
            normalised_gap = (ankle_y - hip_y) / box_h
            # Gap < 0.18 of box height means hips have descended near ankles
            is_crouching = normalised_gap < 0.18

        # Camera proximity check
        l_knee = kp(13); r_knee = kp(14)
        lower_limbs_visible = any([l_knee, r_knee, l_ankle, r_ankle])

        if not lower_limbs_visible and is_horizontal:
            is_horizontal = False   # box-shape unreliable without lower body

        # H6: HAND RAISE GESTURE
        # If a wrist is above the nose, the person is signalling they
        # are okay. Cancels and resets active alerts.
        # Keypoints: 0=Nose, 9=L-Wrist, 10=R-Wrist
        is_hand_raised = False

        nose     = kp(0)
        l_wrist  = kp(9)
        r_wrist  = kp(10)

        if nose:
            # Lower Y in image coords = higher on screen
            if l_wrist and l_wrist[1] < nose[1]:
                is_hand_raised = True
            if r_wrist and r_wrist[1] < nose[1]:
                is_hand_raised = True

        # Gesture confirmation
        if is_hand_raised:
            state['hand_raised_frames'] += 1
        else:
            state['hand_raised_frames'] = 0

        # Only confirm raise if it persists for X frames
        is_hand_raised_confirmed = state['hand_raised_frames'] >= GESTURE_CONFIRM_FRAMES

        if is_hand_raised_confirmed and state['status'] != 'normal':
            print(f"[ID {track_id}] 👋 Sustained hand-raise detected — cancelling.")

        # Fall Decision Logic
        primary_count = sum([is_falling_fast, is_spine_tilted, is_horizontal])
        suppressed    = is_crouching or is_hand_raised_confirmed

        active_fall = (primary_count >= 2) and not suppressed

        # Latch: once in fallen_waiting/emergency, stay there while
        # still horizontal (natural ground-dwelling after a fall).
        if (state['status'] in ('fallen_waiting', 'emergency')
                and is_horizontal and not is_hand_raised_confirmed):
            active_fall = True

        # STATE MACHINE
        if active_fall:
            # Reset recovery latch if we re-detect the fall
            state['recovery_start_time'] = None

            if state['status'] == 'normal':
                print(f"[ID {track_id}] ⚠️  Potential fall — starting {RECOVERY_TIMEOUT_SECONDS}s timer.")
                state['status']         = 'fallen_waiting'
                state['fall_timestamp'] = current_time

            elif state['status'] == 'fallen_waiting':
                elapsed = current_time - state['fall_timestamp']
                if elapsed >= RECOVERY_TIMEOUT_SECONDS:
                    print(f"[ID {track_id}] 🚨 Recovery timeout expired — TRUE FALL confirmed.")
                    state['status'] = 'emergency'

        else:
            # Person is upright/standing OR cancelled via hand raise
            if state['status'] in ('fallen_waiting', 'emergency', 'inactive'):
                # Recovery latch (prevents oscillation)
                if is_hand_raised_confirmed:
                    # Gestures skip the latch — immediate cancel
                    print(f"[ID {track_id}] ✅ Alert cancelled via hand gesture.")
                    state['status'] = 'normal'
                    state['fall_timestamp'] = None
                else:
                    # Standing up requires the latch
                    if state['recovery_start_time'] is None:
                        state['recovery_start_time'] = current_time
                    
                    elapsed_recovery = current_time - state['recovery_start_time']
                    if elapsed_recovery >= RECOVERY_LATCH_SECONDS:
                        print(f"[ID {track_id}] ✅ Alert cleared (Sustained recovery).")
                        state['status']         = 'normal'
                        state['fall_timestamp'] = None
                        state['recovery_start_time'] = None
            else:
                state['recovery_start_time'] = None

        # Inactivity failsafe
        if (is_horizontal
                and state['status'] not in ('emergency', 'inactive')
                and not is_hand_raised_confirmed):
            inactive_for = current_time - state['last_active_time']
            if inactive_for > INACTIVITY_THRESHOLD:
                print(f"[ID {track_id}] 🛌 Horizontal + motionless for "
                      f"{inactive_for:.1f}s — INACTIVITY EMERGENCY.")
                state['status'] = 'inactive'

        # Debug Data
        state['debug'] = {
            'aspect_ratio'    : round(aspect_ratio, 3),
            'spine_angle'     : round(spine_angle_deg, 1) if spine_angle_deg else None,
            'velocity_ema'    : round(state['velocity_ema'] or 0, 4),
            'is_horizontal'   : is_horizontal,
            'is_spine_tilted' : is_spine_tilted,
            'is_falling_fast' : is_falling_fast,
            'is_crouching'    : is_crouching,
            'is_hand_raised'  : is_hand_raised,
            'primary_count'   : primary_count,
            'status'          : state['status'],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cleanup_stale_tracks(self, current_time: float) -> None:
        stale = [tid for tid, s in self.person_states.items()
                 if current_time - s['last_seen'] > STALE_TRACK_TIMEOUT]
        for tid in stale:
            del self.person_states[tid]

    def _update_global_prediction(self) -> None:
        any_emergency = any(
            s['status'] in ('emergency', 'inactive')
            for s in self.person_states.values()
        )
        self._global_prediction = "fall" if any_emergency else "normal"

    def draw_landmarks(self, frame: np.ndarray, results) -> None:
        """Overlay YOLOv8 skeleton + status badges on frame (in-place)."""
        if not results or len(results) == 0:
            return

        annotated = results[0].plot()
        frame[:] = annotated[:]

        # Draw per-person status badge
        if results[0].boxes.id is None:
            return

        for i, box in enumerate(results[0].boxes):
            tid = int(box.id.item())
            if tid not in self.person_states:
                continue
            state  = self.person_states[tid]
            status = state['status']

            x1 = int(box.xyxy[0][0].cpu().numpy())
            y1 = int(box.xyxy[0][1].cpu().numpy())

            colour = {
                'normal'        : (0,  200, 0),
                'fallen_waiting': (0,  165, 255),
                'emergency'     : (0,  0,   255),
                'inactive'      : (0,  0,   200),
            }.get(status, (128, 128, 128))

            import cv2
            label = f"ID{tid}:{status.upper()}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    def reset(self) -> None:
        """
        Clear the global alert lock WITHOUT wiping person states.

        FIX vs original: original reset() called person_states.clear(),
        which caused a 2-frame blind spot and re-detection loop when the
        person was still on the ground after alert escalation.
        Now we only reset status → normal for persons currently in
        'emergency' (user-acknowledged), not for the tracking history.
        """
        self._global_prediction = "normal"
        for state in self.person_states.values():
            if state['status'] in ('emergency', 'inactive'):
                state['status']         = 'normal'
                state['fall_timestamp'] = None
                state['velocity_ema']   = None

    def hard_reset(self) -> None:
        """Full wipe — use only on stream restart or camera change."""
        self._global_prediction = "normal"
        self.person_states.clear()


# ------------------------------------------------------------------
# State factory

def _create_empty_state() -> dict:
    now = time.time()
    return {
        'status'          : 'normal',
        'fall_timestamp'  : None,
        'prev_torso_y'    : None,
        'prev_time'       : now,
        'velocity_ema'    : None,
        'last_seen'       : now,
        'last_active_time': now,
        'torso_sh'        : None,
        'torso_hp'        : None,
        'hand_raised_frames': 0,
        'recovery_start_time': None,
        'debug'           : {},
    }