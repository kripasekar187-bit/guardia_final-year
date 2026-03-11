import cv2
import numpy as np

from guardia.config import NIGHT_BRIGHTNESS_THRESHOLD, NIGHT_ALPHA, NIGHT_BETA


def adjust_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < NIGHT_BRIGHTNESS_THRESHOLD:
        frame = cv2.convertScaleAbs(frame, alpha=NIGHT_ALPHA, beta=NIGHT_BETA)
        cv2.putText(frame, "NIGHT MODE ACTIVE", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return frame
