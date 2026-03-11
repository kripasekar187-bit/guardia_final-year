"""
guardia/alerts.py
-----------------
Everything related to alerting: voice, alarm beep, image capture, and log file.

WHY THREADING FOR TTS?
  pyttsx3's runAndWait() blocks until the speech finishes.
  If we called it on the main thread, the video would freeze while talking.
  Running it in a daemon thread keeps the camera loop smooth.
"""

import os
import time
import threading
import winsound

import cv2
import pyttsx3

from guardia.config import TTS_RATE, ALERTS_LOG, FALL_IMAGE, LOGS_DIR

# ── Text-to-Speech engine (initialised once, reused) ──────────────────────
_engine = pyttsx3.init()
_engine.setProperty('rate', TTS_RATE)


def speak_async(text):
    """Speak 'text' on a background thread so the main loop isn't blocked."""
    def _run():
        _engine.say(text)
        _engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


# ── Alarm ──────────────────────────────────────────────────────────────────
def play_alarm():
    """Play 5 beeps using the Windows system speaker."""
    for _ in range(5):
        winsound.Beep(1200, 800)


# ── Logging ────────────────────────────────────────────────────────────────
def log_alert(frame):
    """
    Save a snapshot of the frame and append a timestamped entry to alerts.txt.
    Creates the logs/ directory automatically if it doesn't exist yet.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    cv2.imwrite(FALL_IMAGE, frame)
    with open(ALERTS_LOG, "a") as f:
        f.write(f"Emergency detected at {time.ctime()}\n")
