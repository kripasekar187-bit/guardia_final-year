import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading
import winsound

# -------------------- VOICE ENGINE --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# -------------------- ALARM FUNCTION --------------------
def play_alarm():
    for i in range(5):
        winsound.Beep(1200, 800)

# -------------------- NIGHT MODE FUNCTION --------------------
def adjust_brightness(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 60:   # dark environment

        alpha = 1.8   # contrast
        beta = 40     # brightness

        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        cv2.putText(frame,
                    "NIGHT MODE ACTIVE",
                    (50,150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2)

    return frame


# -------------------- MEDIAPIPE SETUP --------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

# -------------------- VARIABLES --------------------
prev_head_y = None
fall_detected = False
alert_triggered = False
alert_time = 0

last_movement_time = time.time()

INACTIVITY_THRESHOLD = 15
FALL_THRESHOLD = 0.12

print("GUARDIA Monitoring Started")

# -------------------- MAIN LOOP --------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Apply Night Mode
    frame = adjust_brightness(frame)

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # -------- FALL DETECTION --------
        head_y = landmarks[0].y

        if prev_head_y is not None:
            drop = head_y - prev_head_y
            if drop > FALL_THRESHOLD:
                fall_detected = True

        prev_head_y = head_y

        # -------- MOVEMENT DETECTION --------
        pose_array = np.array([[lm.x, lm.y] for lm in landmarks])
        movement = np.linalg.norm(pose_array - pose_array.mean(axis=0))

        if movement > 0.02:
            last_movement_time = time.time()

        inactivity_time = time.time() - last_movement_time

        # -------- EMERGENCY DETECT --------
        if (fall_detected or inactivity_time > INACTIVITY_THRESHOLD) and not alert_triggered:

            print("⚠ EMERGENCY DETECTED")
            speak_async("Are you okay?")

            alert_triggered = True
            alert_time = time.time()

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # -------- SHOW EMERGENCY TEXT --------
    if alert_triggered:
        cv2.putText(frame,
                    "EMERGENCY DETECTED",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

        cv2.putText(frame,
                    "PRESS Y IF OK",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    cv2.imshow("GUARDIA - Elderly Monitoring", frame)

    # -------- KEYBOARD INPUT --------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('y') and alert_triggered:

        print("User responded. Alert cancelled")
        speak_async("Okay")

        alert_triggered = False
        fall_detected = False
        last_movement_time = time.time()

    # -------- NO RESPONSE CASE --------
    if alert_triggered and (time.time() - alert_time > 7):

        print("🚨 ALERT: Caregiver Notified")

        speak_async("Emergency detected. Please help.")

        play_alarm()

        cv2.imwrite("fall_event.jpg", frame)

        with open("alerts.txt", "a") as f:
            f.write("Emergency detected at " + time.ctime() + "\n")

        alert_triggered = False
        fall_detected = False
        last_movement_time = time.time()

    if key == ord('q'):
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()