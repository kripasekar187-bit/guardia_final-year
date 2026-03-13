"""
guardia/main.py - Main entry point and real-time dashboard.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['GLOG_minloglevel']       = '2'

import time
import cv2
import numpy as np

from guardia.config   import (
    CAMERA_INDEX,
    ALERT_RESPONSE_TIMEOUT,
)
from guardia.detector  import GuardiaDetector
from guardia.night_mode import adjust_brightness
from guardia.alerts    import (
    alert_speak, escalation_speak, ok_speak,
    stop_alarm, log_alert,
)

# --- UI Constants ---

# Colours  (BGR)
C_GREEN      = (50,  220, 80)
C_ORANGE     = (0,   165, 255)
C_RED        = (0,   50,  255)
C_WHITE      = (255, 255, 255)
C_BLACK      = (0,   0,   0)
C_DARK       = (20,  20,  20)
C_PANEL_BG   = (15,  15,  15)    # HUD panel background
C_ACCENT     = (0,   200, 255)   # teal accent for labels

FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL   = 0.42
FONT_MED     = 0.58
FONT_LARGE   = 0.9
FONT_BOLD    = cv2.FONT_HERSHEY_DUPLEX

HUD_WIDTH    = 230   # pixels wide for left HUD panel
PANEL_ALPHA  = 0.72  # transparency of HUD overlay (0=invisible, 1=opaque)


# --- Drawing Helpers ---

def _overlay_rect(frame, x1, y1, x2, y2, colour, alpha):
    """Draw a semi-transparent filled rectangle."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    rect = np.full_like(roi, colour)
    cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _text(frame, txt, x, y, colour=C_WHITE, scale=FONT_SMALL,
          thickness=1, font=FONT):
    cv2.putText(frame, txt, (x, y), font, scale, C_BLACK, thickness + 2)
    cv2.putText(frame, txt, (x, y), font, scale, colour, thickness)


def _progress_bar(frame, x, y, w, h, ratio, fg_colour, bg_colour=C_DARK):
    """Draw a filled progress bar.  ratio ∈ [0, 1]."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_colour, -1)
    filled = max(0, min(int(w * ratio), w))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), fg_colour, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_WHITE, 1)


# --- HUD Rendering ---

def draw_hud(frame, detector: GuardiaDetector, fps: float,
             alert_triggered: bool, alert_time: float,
             night_mode: bool, current_time: float):
    h, w = frame.shape[:2]

    # HUD panel
    _overlay_rect(frame, 0, 0, HUD_WIDTH, h, C_PANEL_BG, PANEL_ALPHA)
    cv2.line(frame, (HUD_WIDTH, 0), (HUD_WIDTH, h), C_ACCENT, 1)

    y = 22

    # Logo / title
    _text(frame, "GUARDIA", 8, y, C_ACCENT, 0.65, 2, FONT_BOLD)
    y += 20
    cv2.line(frame, (6, y), (HUD_WIDTH - 6, y), C_ACCENT, 1)
    y += 14

    # System status badge
    if alert_triggered or detector.emergency:
        status_txt = "EMERGENCY"
        status_col = C_RED
    elif any(s['status'] == 'fallen_waiting'
             for s in detector.person_states.values()):
        status_txt = "POSSIBLE FALL"
        status_col = C_ORANGE
    else:
        status_txt = "MONITORING"
        status_col = C_GREEN

    _text(frame, "STATUS", 8, y, C_ACCENT, FONT_SMALL)
    y += 16
    _text(frame, status_txt, 8, y, status_col, FONT_MED, 2, FONT_BOLD)
    y += 22
    cv2.line(frame, (6, y), (HUD_WIDTH - 6, y), (50, 50, 50), 1)
    y += 10

    # Persons tracked
    n_persons = len(detector.person_states)
    _text(frame, f"TRACKING  {n_persons} person{'s' if n_persons != 1 else ''}",
          8, y, C_WHITE, FONT_SMALL)
    y += 18

    # Per-person signal breakdown
    for tid, state in detector.person_states.items():
        dbg = state.get('debug', {})
        if not dbg:
            continue

        st = state['status']
        pid_col = {
            'normal'        : C_GREEN,
            'fallen_waiting': C_ORANGE,
            'emergency'     : C_RED,
            'inactive'      : C_RED,
        }.get(st, C_WHITE)

        _text(frame, f"ID {tid} — {st.upper()}", 8, y, pid_col, FONT_SMALL, 1)
        y += 14

        # Signal indicators
        signals = [
            ("VEL", dbg.get('is_falling_fast', False)),
            ("SPL", dbg.get('is_spine_tilted', False)),
            ("A/R", dbg.get('is_horizontal',   False)),
        ]
        sx = 10
        for label, active in signals:
            col  = C_RED if active else (80, 80, 80)
            cv2.rectangle(frame, (sx, y - 10), (sx + 32, y + 2), col, -1)
            cv2.rectangle(frame, (sx, y - 10), (sx + 32, y + 2), C_WHITE, 1)
            _text(frame, label, sx + 3, y, C_WHITE, 0.35, 1)
            sx += 36
        y += 14

        # Recovery countdown bar (fallen_waiting only)
        if st == 'fallen_waiting' and state.get('fall_timestamp'):
            from guardia.config import RECOVERY_TIMEOUT_SECONDS
            elapsed = current_time - state['fall_timestamp']
            ratio   = min(elapsed / RECOVERY_TIMEOUT_SECONDS, 1.0)
            _text(frame, "RECOVERY TIMER", 8, y, C_ORANGE, 0.35)
            y += 12
            _progress_bar(frame, 8, y, HUD_WIDTH - 16, 8, ratio, C_ORANGE)
            y += 14

        y += 4
        if y > h - 60:
            break   # ran out of panel space

    # Status bar
    bar_y = h - 26
    _overlay_rect(frame, 0, bar_y, w, h, C_PANEL_BG, PANEL_ALPHA)
    cv2.line(frame, (0, bar_y), (w, bar_y), C_ACCENT, 1)

    _text(frame, f"FPS {fps:.1f}", 8, h - 8, C_WHITE, FONT_SMALL)

    mode_txt = "NIGHT MODE ON" if night_mode else "NORMAL LIGHT"
    mode_col = C_ORANGE if night_mode else C_GREEN
    _text(frame, mode_txt, 90, h - 8, mode_col, FONT_SMALL)

    hints = "[Y] I'm OK    [Q] Quit"
    tw    = cv2.getTextSize(hints, FONT, FONT_SMALL, 1)[0][0]
    _text(frame, hints, w - tw - 10, h - 8, (160, 160, 160), FONT_SMALL)

    # Emergency banner
    if alert_triggered or detector.emergency:
        bw, bh = 420, 90
        bx = (w - bw) // 2
        by = (h - bh) // 2 - 30

        # Pulsing border (alternate every ~0.5 s)
        pulse = int(current_time * 2) % 2 == 0
        border_col = C_RED if pulse else C_ORANGE

        _overlay_rect(frame, bx, by, bx + bw, by + bh, (0, 0, 0), 0.78)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_col, 3)

        _text(frame, "! EMERGENCY DETECTED !",
              bx + 18, by + 32, C_RED, FONT_LARGE, 2, FONT_BOLD)
        _text(frame, "PRESS  Y  —  I AM OKAY",
              bx + 30, by + 60, C_WHITE, FONT_MED, 1)

        # Response countdown bar
        if alert_time > 0:
            elapsed = current_time - alert_time
            ratio   = max(0.0, 1.0 - elapsed / ALERT_RESPONSE_TIMEOUT)
            bar_col = C_GREEN if ratio > 0.5 else (C_ORANGE if ratio > 0.2 else C_RED)
            _progress_bar(frame, bx + 10, by + 72, bw - 20, 10, ratio, bar_col)
            secs_left = max(0, int(ALERT_RESPONSE_TIMEOUT - elapsed) + 1)
            _text(frame, f"Auto-escalate in {secs_left}s",
                  bx + 10, by + 90, (180, 180, 180), 0.38)


# --- Main loop ---

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[GUARDIA] Camera not accessible.")
        return

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector          = GuardiaDetector()
    alert_triggered   = False
    alert_time        = 0.0
    escalated         = False        # FIX: track escalation separately
    night_mode_active = False

    # FPS calculation
    fps_counter  = 0
    fps_display  = 0.0
    fps_timer    = time.time()
    saved_image_path = None

    print("[GUARDIA] Started — press Q or Esc to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[GUARDIA] Camera read failed.")
            break

        current_time = time.time()

        # Pre-processing
        _result = adjust_brightness(frame)
        if isinstance(_result, tuple):
            frame, is_night = _result
        else:
            frame   = _result
            is_night = False
        night_mode_active = is_night
        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detection
        results = detector.process(frame_rgb)
        if results and len(results) > 0:
            detector.update(results)
            detector.draw_landmarks(frame, results)

        # Alert logic

        # New emergency — trigger initial TTS once
        if detector.emergency and not alert_triggered and not escalated:
            print("[GUARDIA] EMERGENCY DETECTED")
            alert_speak()
            alert_triggered = True
            alert_time      = current_time
            escalated       = False
            saved_image_path = log_alert(frame, metadata={
                'persons': len(detector.person_states),
                'states' : {tid: s['status'] for tid, s in detector.person_states.items()}
            })

        # Auto-cancel: detector cleared itself (person recovered / hand raise)
        if (alert_triggered or escalated) and not detector.emergency:
            print("[GUARDIA] Detector cleared — cancelling alert.")
            stop_alarm()
            alert_triggered  = False
            escalated        = False
            saved_image_path = None

        # Escalation timeout
        if alert_triggered and not escalated:
            if current_time - alert_time > ALERT_RESPONSE_TIMEOUT:
                print("[GUARDIA] No response — ESCALATING.")
                escalation_speak(saved_image_path)
                escalated       = True
                alert_triggered = False  # banner gone, alarm continues

        # HUD drawing
        draw_hud(frame, detector, fps_display,
                 alert_triggered, alert_time,
                 night_mode_active, current_time)

        # FPS calculation
        fps_counter += 1
        if current_time - fps_timer >= 1.0:
            fps_display  = fps_counter / (current_time - fps_timer)
            fps_counter  = 0
            fps_timer    = current_time

        cv2.imshow("GUARDIA — Elderly Fall Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # Keyboard input
        if key == ord('y') or key == ord('Y'):
            if alert_triggered or escalated:
                print("[GUARDIA] User confirmed OK.")
                ok_speak()
                stop_alarm()
                alert_triggered = False
                escalated       = False
                saved_image_path = None
                detector.reset()

        # Q / Esc = quit
        if key in (ord('q'), ord('Q'), 27):
            stop_alarm()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[GUARDIA] Stopped.")


if __name__ == "__main__":
    main()