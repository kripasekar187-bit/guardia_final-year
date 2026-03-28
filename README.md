# Guardia

A real-time fall detection system for elderly care and solo-living safety. Built as a final year project, Guardia uses pose estimation to watch over people who live alone — and acts fast when something goes wrong.

---

## What It Does

Guardia runs on a standard webcam and processes video in real time. When it detects a fall, it:

1. Speaks an "Are you okay?" prompt through the device speakers
2. Sounds an audio alarm if there's no response
3. Sends an SMS and WhatsApp alert to a designated emergency contact
4. Uploads a snapshot of the fall to cloud storage for reference

The whole pipeline — from fall to SMS delivery — takes around 12 seconds.

---

## How It Works

Rather than training a custom model from scratch, Guardia uses YOLOv8n-pose (a pre-trained skeleton detection model) and applies multi-signal heuristics on top of the output:

- **Velocity spike** — sudden downward acceleration of the body center
- **Horizontal posture** — aspect ratio of the bounding box flips when someone falls
- **Spine angle** — skeleton orientation crosses a threshold toward horizontal

A fall is only flagged when at least two of these three signals trigger simultaneously. This cuts false positives from simple actions like deep bending or sitting cross-legged.

Each person in frame is tracked independently. There's a 3.5-second grace period after a fall is detected — if the person gets back up on their own, no alert is sent. If they raise a hand and hold it up for a sustained period, they can cancel the alert themselves.

---

## Results

Tested on 75 video clips across 7 categories with 3 subjects:

| Metric | Score |
|---|---|
| Accuracy | 91.0% |
| Precision | 81.0% |
| Recall | 85.0% |
| False positive rate | 7.3% |

Performance drops in backlit conditions (79%) and on slow sideways falls near walls. Full breakdown in [EVALUATION.md](EVALUATION.md).

---

## Setup

**Requirements:** Python 3.9+, a webcam, and a Twilio account for SMS alerts.

```bash
git clone <repo-url>
cd guardia_final-year

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Open `guardia/config.py` and fill in your emergency contact number and Twilio credentials. Then run:

```bash
python main.py
```

The YOLOv8n-pose model downloads automatically on first run (~6MB). You can also drop a pre-downloaded `yolov8n-pose.pt` into the `models/` folder.

---

## Key Configuration

All thresholds live in `guardia/config.py`. The defaults work well in most indoor conditions, but you can tune them:

```python
FALLING_VELOCITY_THRESHOLD = 0.12    # Higher = less sensitive to velocity spikes
RECOVERY_TIMEOUT_SECONDS   = 3.5     # Grace period before escalation
INACTIVITY_THRESHOLD       = 8       # Seconds motionless before alert triggers
ENABLE_SMS                 = True    # Toggle Twilio alerts on/off
```

---

## Project Structure

```
guardia_final-year/
├── main.py                   # Entry point and webcam loop
├── requirements.txt
├── guardia/
│   ├── detector.py           # Core heuristics and per-person state machine
│   ├── alerts.py             # TTS, audio, SMS, WhatsApp, image upload
│   ├── config.py             # Thresholds and credentials
│   └── night_mode.py         # Low-light brightness enhancement
├── guardia-android/          # Android app (Kotlin + TFLite)
├── models/                   # YOLOv8n-pose weights
├── logs/                     # Alert history and fall snapshots
├── EVALUATION.md             # Full test report
└── ANDROID_INTEGRATION_PLAN.md
```

---

## Android

There's a companion Android app under `guardia-android/` that runs inference on-device using TensorFlow Lite and CameraX. It's functional but still being polished — setup steps are in [ANDROID_INTEGRATION_PLAN.md](ANDROID_INTEGRATION_PLAN.md).

---

## Limitations

- Works best with a camera mounted at roughly 1.5m height at a 45° downward angle
- Detection degrades in backlit scenes or heavy occlusion
- Slow sideways falls (where the person slides rather than drops) are sometimes missed
- Multi-person tracking is supported but hasn't been stress-tested beyond 2–3 people in frame

---

## Stack

Python · OpenCV · YOLOv8 (Ultralytics) · pyttsx3 · Twilio · ImgBB · Kotlin · TensorFlow Lite · CameraX

---

## Authors

Nipun Sujesh · Kripa Sekar
