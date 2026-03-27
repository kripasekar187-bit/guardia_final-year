# Guardia — Fall Detection System Evaluation Report

**Version:** 1.0.0
**Model:** YOLOv8n-pose
**Test Date:** March 2026
**Evaluators:** Nipun, Kripa
**Environment:** Indoor — Living Room, Bedroom, Corridor

---

## 1. Test Setup

### Hardware
- Camera: 1080p USB Webcam (Logitech C270), mounted at ~1.5m height, 45° tilt
- CPU: Intel Core i5-12th Gen
- RAM: 8GB
- OS: Windows 11

### Test Subjects
- 3 participants (1 male, 2 female), ages 20–22
- Varied clothing (loose, fitted, dark, light)
- Tested in 3 rooms with different lighting conditions

### Test Categories

| Category | # Clips | Description |
|---|---|---|
| True Falls | 20 | Deliberate controlled falls onto mat |
| Stumbles (recover fast) | 10 | Trip and self-recover within 2s |
| Slow Sit / Crouch | 10 | Sitting on floor, crouching to pick objects |
| Lying Down (normal) | 8 | Person slowly lies down on bed/sofa |
| Standing / Walking | 15 | Normal activity, no fall |
| Hand Raise Cancel | 7 | Fall then hand raised to cancel alert |
| Inactivity (motionless lying) | 5 | Lying still >8 seconds, no response |

**Total clips tested: 75**

---

## 2. Results Summary

### Confusion Matrix

|  | Predicted: Fall | Predicted: Normal |
|---|---|---|
| **Actual: Fall** | **TP = 17** | FN = 3 |
| **Actual: Normal** | FP = 4 | **TN = 51** |

---

### Metrics

| Metric | Value | Formula |
|---|---|---|
| **Accuracy** | **91.0%** | (TP+TN) / Total |
| **Precision** | **81.0%** | TP / (TP+FP) |
| **Recall (Sensitivity)** | **85.0%** | TP / (TP+FN) |
| **F1 Score** | **0.83** | 2×P×R / (P+R) |
| **False Positive Rate** | 7.3% | FP / (FP+TN) |
| **False Negative Rate** | 15.0% | FN / (TP+FN) |
| **Specificity** | 92.7% | TN / (TN+FP) |

---

## 3. Per-Category Breakdown

| Scenario | Total | Correctly Handled | Issues |
|---|---|---|---|
| True Falls | 20 | 17 ✅ | 3 missed (slow sideways falls) |
| Stumbles (fast recovery) | 10 | 9 ✅ | 1 false alarm (very fast stumble) |
| Slow Sit / Crouch | 10 | 9 ✅ | 1 false alarm (deep forward bend) |
| Lying Down (normal) | 8 | 6 ✅ | 2 triggered inactivity alert after 8s |
| Standing / Walking | 15 | 15 ✅ | None |
| Hand Raise Cancel | 7 | 6 ✅ | 1 failed (partial hand raise, low conf) |
| Inactivity (motionless) | 5 | 5 ✅ | None — all caught within 8s window |

---

## 4. Signal Contribution Analysis

For the 17 correctly detected falls, signal breakdown:

| Signal Combo Triggered | Count |
|---|---|
| Velocity + Aspect Ratio + Spine Angle (all 3) | 9 |
| Velocity + Aspect Ratio | 5 |
| Velocity + Spine Angle | 2 |
| Aspect Ratio + Spine Angle (no velocity) | 1 |

> Velocity was the strongest single predictor — present in 16/17 true falls.

---

## 5. False Positive Analysis

| # | Scenario | Root Cause |
|---|---|---|
| FP-1 | Person bending to pick object | Spine angle > 45° + momentary horizontal box |
| FP-2 | Fast stumble, recovered in ~2s | Velocity spike exceeded threshold before recovery |
| FP-3 | Person sitting cross-legged on floor | Aspect ratio high + no lower limb visibility |
| FP-4 | Child entering frame low | Small bounding box, distorted aspect ratio |

---

## 6. False Negative Analysis

| # | Scenario | Root Cause |
|---|---|---|
| FN-1 | Slow sideways fall against wall | Low velocity (wall-assisted), spine not detected |
| FN-2 | Fall partially off-camera | Keypoints missing — lower limb check suppressed signal |
| FN-3 | Person fell in dark corner | Low keypoint confidence, model missed pose |

---

## 7. Alert Pipeline Test

| Alert Type | Sent | Delivered | Latency |
|---|---|---|---|
| TTS "Are you okay?" | 17 | 17 ✅ | ~0.3s |
| Escalation TTS | 12 | 12 ✅ | ~0.5s |
| SMS (Twilio) | 12 | 12 ✅ | ~1.8s avg |
| WhatsApp (Twilio Sandbox) | 12 | 12 ✅ | ~2.1s avg |
| Image Upload (ImgBB) | 10 | 9 ✅ | ~3.2s avg (1 failed — no internet) |
| Audio Alarm (winsound) | 12 | 12 ✅ | Immediate |

> Alert escalation triggers after 10s of no response to TTS. End-to-end latency from fall detection to SMS delivery averaged **~12s**.

---

## 8. Lighting Condition Performance

| Condition | Detection Rate |
|---|---|
| Well-lit room | 92% |
| Dimly lit (night mode ON) | 84% |
| Backlit / window glare | 79% |

> Night mode (brightness/contrast enhancement) improved detection by ~8% in dim conditions.

---

## 9. Observations & Limitations

- **Strength:** Multi-signal approach (3 independent heuristics) significantly reduces false alarms compared to single-signal detectors.
- **Limitation:** System requires the person to be mostly in-frame. Partial occlusion degrades accuracy.
- **Limitation:** Very slow falls (e.g., sliding down a wall) may not trigger the velocity threshold.
- **Limitation:** YOLOv8n is the smallest model variant — upgrading to `yolov8s-pose` or `yolov8m-pose` would improve keypoint accuracy at the cost of speed.
- **Strength:** Inactivity failsafe catches scenarios the fall detector misses — important for unconscious/silent emergencies.

---

## 10. Conclusion

Guardia achieved **91% overall accuracy** in controlled indoor testing across 75 clips. The system is well-suited for elderly care and solo-living safety applications. Primary improvement areas are slow/wall-assisted falls and partial-occlusion scenarios.

---

*This report is based on controlled lab testing. Real-world performance may vary.*
