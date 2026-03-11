"""
guardia/night_mode.py
---------------------
Adjusts the camera frame for dark environments.

HOW IT WORKS:
  We compute the mean pixel brightness of a grayscale version of the frame.
  If it's below the threshold, we boost contrast (alpha) and brightness (beta)
  using OpenCV's convertScaleAbs, which applies:  output = alpha * pixel + beta
"""

import cv2
import numpy as np

from guardia.config import NIGHT_BRIGHTNESS_THRESHOLD, NIGHT_ALPHA, NIGHT_BETA


def adjust_brightness(frame):
    """
    Returns the frame unchanged in normal light.
    In dark environments, boosts contrast + brightness and shows an overlay label.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < NIGHT_BRIGHTNESS_THRESHOLD:
        frame = cv2.convertScaleAbs(frame, alpha=NIGHT_ALPHA, beta=NIGHT_BETA)
        cv2.putText(
            frame,
            "NIGHT MODE ACTIVE",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    return frame
