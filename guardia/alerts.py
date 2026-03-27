"""
guardia/alerts.py - Emergency alert outputs (Audio, TTS, SMS, WhatsApp).
"""

import os
import math
import struct
import threading
import time
import wave
import tempfile
from typing import Optional

import cv2
import requests
import base64

from guardia.config import (
    TTS_RATE,
    TTS_ALERT_MESSAGE,
    TTS_ESCALATION_MESSAGE,
    TTS_OK_MESSAGE,
    ALARM_BEEP_COUNT,
    ALARM_BEEP_FREQ,
    ALARM_BEEP_MS,
    ALERTS_LOG,
    FALL_IMAGE,
    LOGS_DIR,
    ENABLE_SMS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER,
    EMERGENCY_CONTACT_NUMBER,
    IMGBB_API_KEY,
    ENABLE_WHATSAPP,
    WHATSAPP_TO_NUMBERS,
    WHATSAPP_FROM_NUMBER,
)

# --- Image Uploading ---

def upload_image(file_path: str) -> str:
    """Upload a local image to ImgBB and return the public URL."""
    print(f"[GUARDIA Uploader] Starting upload for: {file_path}")
    if not IMGBB_API_KEY or not os.path.exists(file_path):
        print(f"[GUARDIA Uploader] Aborting: Key missing or file not found.")
        return None

    try:
        with open(file_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key": IMGBB_API_KEY,
                "image": base64.b64encode(file.read()),
            }
            response = requests.post(url, payload)
            response.raise_for_status()
            data = response.json()
            return data["data"]["url"]
    except Exception as e:
        print(f"[GUARDIA Uploader] Failed to upload image: {e}")
        return None

# --- Alert Delivery ---

def send_external_alert(message_body: str, media_url: str = None) -> None:
    """Send alert via SMS and/or WhatsApp."""
    if not (ENABLE_SMS or ENABLE_WHATSAPP):
        return

    def _run():
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # Send SMS
            if ENABLE_SMS:
                client.messages.create(
                    body=message_body,
                    from_=TWILIO_FROM_NUMBER,
                    to=EMERGENCY_CONTACT_NUMBER,
                    media_url=[media_url] if media_url else None
                )
                print(f"[GUARDIA SMS] Text alert sent to {EMERGENCY_CONTACT_NUMBER}")

            # Send WhatsApp
            if ENABLE_WHATSAPP:
                for number in WHATSAPP_TO_NUMBERS:
                    try:
                        client.messages.create(
                            body=message_body,
                            from_=WHATSAPP_FROM_NUMBER,
                            to=number,
                            media_url=[media_url] if media_url else None
                        )
                        print(f"[GUARDIA WhatsApp] Alert sent to {number}")
                    except Exception as whatsapp_e:
                        print(f"[GUARDIA WhatsApp error] Failed sending to {number}: {whatsapp_e}")

        except Exception as e:
            print(f"[GUARDIA External Alert error] {e}")

    threading.Thread(target=_run, daemon=True).start()

# --- TTS Engine ---

_tts_lock   = threading.Lock()    # serialises all pyttsx3 calls
_tts_engine = None

def _init_tts():
    global _tts_engine
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', TTS_RATE)
        print("[GUARDIA alerts] TTS engine initialised.")
    except Exception as e:
        print(f"[GUARDIA alerts] TTS unavailable ({e}). Running in silent mode.")
        _tts_engine = None

_init_tts()


def speak_async(text: str) -> None:
    """Speak `text` in a background thread.  Thread-safe."""
    if _tts_engine is None:
        print(f"[GUARDIA TTS-silent] {text}")
        return

    def _run():
        with _tts_lock:
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            except Exception as e:
                print(f"[GUARDIA TTS error] {e}")

    threading.Thread(target=_run, daemon=True).start()


# --- Audio Alarm ---

_alarm_stop_event = threading.Event()


def _beep_winsound(freq: int, duration_ms: int) -> bool:
    """Try winsound.Beep().  Returns True on success."""
    try:
        import winsound
        winsound.Beep(freq, duration_ms)
        return True
    except Exception:
        return False


def _beep_sounddevice(freq: int, duration_ms: int) -> bool:
    """Sine-wave beep via sounddevice."""
    try:
        import numpy as np
        import sounddevice as sd
        sample_rate = 44100
        duration_s  = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        wave_data = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sd.play(wave_data, samplerate=sample_rate)
        sd.wait()
        return True
    except Exception:
        return False


def _beep_terminal() -> bool:
    """Last resort: terminal bell character."""
    try:
        print('\a', end='', flush=True)
        return True
    except Exception:
        return False


def _beep(freq: int = ALARM_BEEP_FREQ, duration_ms: int = ALARM_BEEP_MS) -> None:
    """Single beep using the best available method."""
    if _beep_winsound(freq, duration_ms):
        return
    if _beep_sounddevice(freq, duration_ms):
        return
    _beep_terminal()


def play_alarm() -> None:
    """Background alarm loop."""
    _alarm_stop_event.clear()

    def _run():
        for _ in range(ALARM_BEEP_COUNT):
            if _alarm_stop_event.is_set():
                break
            _beep()
            # Short gap between beeps so they are perceptually distinct
            time.sleep(0.15)

    threading.Thread(target=_run, daemon=True).start()


def stop_alarm() -> None:
    """Signal the alarm loop to stop after the current beep."""
    _alarm_stop_event.set()


# --- Logging ---

def log_alert(frame, metadata: Optional[dict] = None) -> Optional[str]:
    """Log alert event and save snapshot."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Save frame snapshot
    try:
        cv2.imwrite(FALL_IMAGE, frame)
    except Exception as e:
        print(f"[GUARDIA log] Could not save fall image: {e}")
        return None

    # Build log line
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] EMERGENCY DETECTED"
    if metadata:
        details = "  |  ".join(f"{k}={v}" for k, v in metadata.items())
        line += f"  —  {details}"
    line += "\n"

    try:
        with open(ALERTS_LOG, "a") as f:
            f.write(line)
        print(f"[GUARDIA log] {line.strip()}")
        return FALL_IMAGE
    except Exception as e:
        print(f"[GUARDIA log] Could not write alert log: {e}")
        return None


# --- Helpers ---

def alert_speak() -> None:
    """Speak the initial 'Are you okay?' message."""
    speak_async(TTS_ALERT_MESSAGE)


def escalation_speak(image_path: str = None) -> None:
    """Speak escalation, upload image, and send text/WhatsApp alerts."""
    speak_async(TTS_ESCALATION_MESSAGE)
    
    def _background_alert():
        media_url = None
        if image_path:
            print("[GUARDIA] Uploading fall image to cloud...")
            media_url = upload_image(image_path)
        
        timestamp = time.strftime("%H:%M:%S")
        msg = (
            f"🚨 GUARDIA EMERGENCY ALERT\n"
            f"---------------------------\n"
            f"A fall was confirmed at {timestamp}.\n"
            f"The user did not respond to local alerts.\n"
        )
        if media_url:
            msg += f"\n📸 View Capture: {media_url}"
        else:
            msg += "\n(Image upload failed or unavailable)"
        
        msg += "\n\nPlease check on them immediately."
        
        send_external_alert(msg, media_url)
        play_alarm()

    threading.Thread(target=_background_alert, daemon=True).start()


def ok_speak() -> None:
    """Speak the cancellation confirmation."""
    speak_async(TTS_OK_MESSAGE)