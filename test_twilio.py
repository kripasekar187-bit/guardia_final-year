from twilio.rest import Client
import sys
import os

# Ensure guardia module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guardia.config import (
    TWILIO_ACCOUNT_SID, 
    TWILIO_AUTH_TOKEN, 
    WHATSAPP_FROM_NUMBER, 
    WHATSAPP_TO_NUMBERS
)

def test_whatsapp():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        for number in WHATSAPP_TO_NUMBERS:
            print(f"Attempting to send WhatsApp message from {WHATSAPP_FROM_NUMBER} to {number}...")
            try:
                msg = client.messages.create(
                    body="🚨 GUARDIA TEST: If you receive this, the Twilio API is working perfectly for multiple numbers! 🚨",
                    from_=WHATSAPP_FROM_NUMBER,
                    to=number
                )
                print(f"✅ Success! Message SID for {number}: {msg.sid}")
            except Exception as e:
                print(f"❌ Failed to send to {number}: {e}")
        print("Check the WhatsApp numbers to see if messages arrived!")
    except Exception as e:
        print(f"❌ Twilio Setup Error: {e}")

if __name__ == "__main__":
    test_whatsapp()
