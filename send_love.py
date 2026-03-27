import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guardia.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, WHATSAPP_FROM_NUMBER
from twilio.rest import Client

def send_love():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body="Fuck u",
            from_=WHATSAPP_FROM_NUMBER,
            to="whatsapp:+918921365016"
        )
        print(f"✅ 'I love u' message sent successfully! SID: {msg.sid}")
    except Exception as e:
        print(f"❌ Failed to send message: {e}")

if __name__ == "__main__":
    send_love()
