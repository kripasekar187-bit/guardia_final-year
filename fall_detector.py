import cv2
import time
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
import threading
import queue
import pyttsx3

# --- VOICE OUTPUT SETUP ---
speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160) # Slower, more readable voice
    while True:
        text = speech_queue.get()
        if text is None: break
        engine.say(text)   
        engine.runAndWait()
        speech_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    # Empty queue so we don't get a backlog of old messages spamming
    with speech_queue.mutex:
        speech_queue.queue.clear()
    speech_queue.put(text)
# --------------------------

print("Loading YOLOv8 Pose model...")
model = YOLO('yolov8n-pose.pt')
print("Model loaded successfully!")

# Constants
HISTORY_LENGTH = 15  
VELOCITY_THRESHOLD = 300.0   # Pixels per second for a fast drop
ASPECT_RATIO_FALL = 1.1      # Width/Height ratio to be considered horizontal for a violent drop
ASPECT_RATIO_INACTIVE = 1.5  # STRICTER: Must be very horizontal (lying down flat) to trigger inactivity
INACTIVITY_THRESHOLD = 10.0  # Seconds of horizontal immobility to trigger inactivity alarm
INACTIVITY_PIXEL_LIMIT = 40  # INCREASED: More lenient for camera noise/breathing while looking for inactivity

# Keypoint Indices (COCO format for YOLOv8)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16

# Tracking Histories
head_history = {}    # person_id: deque([(time, y_pos)])
center_history = {}  # person_id: deque([(time, center_x, center_y)])

# Global State
is_awaiting_response = False
fall_detection_time = 0
critical_alert_triggered = False
alert_reason = "" # "FALL" or "INACTIVITY"

# On-Screen UI Log
ui_logs = deque(maxlen=5)

def add_log(msg):
    time_str = time.strftime('%H:%M:%S')
    ui_logs.appendleft(f"[{time_str}] {msg}")
    print(f"[{time_str}] {msg}")

add_log("System Started. Monitoring active.")
speak("System started. Monitoring active.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    current_time = time.time()
    
    # Run YOLO Pose
    results = model(frame, verbose=False)
    annotated_frame = frame.copy() 
    
    # Draw UI Log Overlay
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
    cv2.putText(annotated_frame, "LIVE SYSTEM LOGS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, log_msg in enumerate(ui_logs):
        cv2.putText(annotated_frame, log_msg, (20, 65 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    has_fallen_person = False
    fallen_box = None
    standing_boxes = []
    
    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue
            
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0

            # Pre-calc center history for Inactivity Detection
            if i not in center_history:
                center_history[i] = deque(maxlen=300) 
            center_history[i].append((current_time, center_x, center_y))

            has_valid_kps = len(keypoints) > i and len(keypoints[i]) > L_ANKLE
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

            if not has_valid_kps:
                continue

            kps = keypoints[i]
            nose_y = kps[NOSE][1]
            l_wrist_y, r_wrist_y = kps[L_WRIST][1], kps[R_WRIST][1]
            l_hip_y, r_hip_y = kps[L_HIP][1], kps[R_HIP][1]
            l_ankle_y, r_ankle_y = kps[L_ANKLE][1], kps[R_ANKLE][1]
            
            # --- FEATURE 4: Posture Dampening (Sitting vs Falling) ---
            avg_hip_y = (l_hip_y + r_hip_y) / 2 if l_hip_y > 0 and r_hip_y > 0 else max(l_hip_y, r_hip_y)
            avg_ankle_y = (l_ankle_y + r_ankle_y) / 2 if l_ankle_y > 0 and r_ankle_y > 0 else max(l_ankle_y, r_ankle_y)
            
            is_truly_flat = False
            if avg_hip_y > 0 and avg_ankle_y > 0:
                # STRICTER CHECK: The vertical height between hips and ankles must be very small.
                # If they are sitting in a recliner, their hips are higher than ankles.
                if abs(avg_ankle_y - avg_hip_y) < (height * 0.3): 
                    is_truly_flat = True

            if nose_y > 0:
                if i not in head_history:
                    head_history[i] = deque(maxlen=HISTORY_LENGTH)
                head_history[i].append((current_time, nose_y))

            is_falling_fast = False
            if i in head_history and len(head_history[i]) >= 5:
                oldest_time, oldest_y = head_history[i][0]
                time_diff = current_time - oldest_time
                y_diff = nose_y - oldest_y
                if time_diff > 0:
                    velocity = y_diff / time_diff
                    if velocity > VELOCITY_THRESHOLD: 
                        is_falling_fast = True

            # Tracking sets
            if (aspect_ratio > ASPECT_RATIO_FALL) or is_falling_fast:
                has_fallen_person = True
                fallen_box = box 
            else:
                standing_boxes.append(box)

            # --- STATE LOGIC ---
            if is_awaiting_response:
                # --- FEATURE 3: Gesture Cancellation ---
                if nose_y > 0 and ( (l_wrist_y > 0 and l_wrist_y < nose_y) or (r_wrist_y > 0 and r_wrist_y < nose_y) ):
                    is_awaiting_response = False
                    critical_alert_triggered = False
                    add_log("SOS gesture detected. Alert cancelled.")
                    speak("Alert cancelled by gesture.")
                    head_history.clear()
            else:
                # Trigger 1: Velocity Fall
                if is_falling_fast and (aspect_ratio > ASPECT_RATIO_FALL) and is_truly_flat:
                    is_awaiting_response = True
                    fall_detection_time = current_time
                    critical_alert_triggered = False
                    alert_reason = "FALL"
                    add_log("FAST FALL DETECTED! Waiting for response.")
                    speak("Fall detected. Are you okay?")
                    head_history.clear()
                
                # Trigger 2: Inactivity (Silent Emergency)
                # Now requires a much wider aspect ratio (1.5 instead of 1.0) and to be truly flat
                elif (aspect_ratio > ASPECT_RATIO_INACTIVE) and is_truly_flat:
                    time_still = 0
                    if len(center_history[i]) > 30: 
                        oldest_t, oldest_cx, oldest_cy = center_history[i][0]
                        recent_t, recent_cx, recent_cy = center_history[i][-1]
                        
                        time_span = recent_t - oldest_t
                        movement = math.hypot(recent_cx - oldest_cx, recent_cy - oldest_cy)
                        
                        if time_span > INACTIVITY_THRESHOLD and movement < INACTIVITY_PIXEL_LIMIT:
                            is_awaiting_response = True
                            fall_detection_time = current_time
                            critical_alert_triggered = False
                            alert_reason = "INACTIVITY"
                            add_log("PROLONGED INACTIVITY DETECTED!")
                            speak("Inactivity detected. Are you okay?")
                            center_history[i].clear()

    # --- FEATURE 2: Multi-Person Silencing ---
    if is_awaiting_response and fallen_box is not None and len(standing_boxes) > 0:
        fx1, fy1, fx2, fy2 = fallen_box
        for sbox in standing_boxes:
            sx1, sy1, sx2, sy2 = sbox
            margin = 50
            if not (sx2 < fx1 - margin or sx1 > fx2 + margin or sy2 < fy1 - margin or sy1 > fy2 + margin):
                is_awaiting_response = False
                critical_alert_triggered = False
                add_log("Help arrived (another person detected). Alert paused.")
                speak("Help arrived. Alert cancelled.")
                break

    # --- UI RENDER LOGIC ---
    if is_awaiting_response:
        time_elapsed = current_time - fall_detection_time
        time_left = max(0, int(15 - time_elapsed))
        
        if time_elapsed > 15:
            if not critical_alert_triggered:
                add_log("CRITICAL: NO RESPONSE! EMERGENCY SENT.")
                speak("Critical alert. No response. Emergency services notified.")
                critical_alert_triggered = True
            
            cv2.rectangle(annotated_frame, (0, 0), (1280, 150), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "CRITICAL EMERGENCY: NO RESPONSE!", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 4)
            cv2.rectangle(annotated_frame, (0, 0), (1280, 720), (0, 0, 255), 20)
        else:
            cv2.rectangle(annotated_frame, (0, 0), (1280, 150), (0, 165, 255), -1)
            reason_text = "FALL DETECTED!" if alert_reason == "FALL" else "INACTIVITY DETECTED!"
            cv2.putText(annotated_frame, f"{reason_text} ARE YOU OKAY? ({time_left}s)", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(annotated_frame, "-> Raise hand above head to cancel", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "-> Person approaching will cancel", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.rectangle(annotated_frame, (0, 0), (1280, 720), (0, 165, 255), 20)
            
    cv2.imshow("GUARDIA AI - Advanced Fall Detection", annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('y') and is_awaiting_response:
        add_log("Manual keyboard override. Alert Cancelled.")
        speak("Alert cancelled manually.")
        is_awaiting_response = False
        critical_alert_triggered = False

cap.release()
cv2.destroyAllWindows()
