import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import time

from guardia.config    import CAMERA_INDEX, ALERT_RESPONSE_TIMEOUT
from guardia.detector  import GuardiaDetector
from guardia.night_mode import adjust_brightness
from guardia.alerts    import speak_async, play_alarm, log_alert


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    detector        = GuardiaDetector()
    alert_triggered = False
    alert_time      = 0.0

    print("GUARDIA started — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = adjust_brightness(frame)
        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = detector.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            detector.update(landmarks)
            detector.draw_landmarks(frame, results.pose_landmarks)

            if detector.emergency and not alert_triggered:
                print("EMERGENCY DETECTED")
                speak_async("Are you okay?")
                alert_triggered = True
                alert_time      = time.time()

        if alert_triggered:
            cv2.putText(frame, "EMERGENCY DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "PRESS Y IF OK", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("GUARDIA", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('y') and alert_triggered:
            print("User responded. Alert cancelled.")
            speak_async("Okay")
            alert_triggered = False
            detector.reset()

        if alert_triggered and (time.time() - alert_time > ALERT_RESPONSE_TIMEOUT):
            print("No response — escalating alert")
            speak_async("Emergency detected. Please help.")
            play_alarm()
            log_alert(frame)
            alert_triggered = False
            detector.reset()

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
