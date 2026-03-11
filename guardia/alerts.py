import os
import time
import threading
import winsound

import cv2
import pyttsx3

from guardia.config import TTS_RATE, ALERTS_LOG, FALL_IMAGE, LOGS_DIR

_engine = pyttsx3.init()
_engine.setProperty('rate', TTS_RATE)


def speak_async(text):
    def _run():
        _engine.say(text)
        _engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


def play_alarm():
    for _ in range(5):
        winsound.Beep(1200, 800)


def log_alert(frame):
    os.makedirs(LOGS_DIR, exist_ok=True)
    cv2.imwrite(FALL_IMAGE, frame)
    with open(ALERTS_LOG, "a") as f:
        f.write(f"Emergency detected at {time.ctime()}\n")
