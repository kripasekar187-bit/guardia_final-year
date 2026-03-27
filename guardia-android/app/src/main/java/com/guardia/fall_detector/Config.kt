package com.guardia.fall_detector

object Config {
    // Alert & Response Timeouts (mapped from config.py)
    const val ALERT_RESPONSE_TIMEOUT_MS = 10000L  // 10 seconds
    const val RECOVERY_TIMEOUT_MS = 3500L         // 3.5 seconds
    
    // SMS Alert (Twilio)
    const val ENABLE_SMS = false
    const val TWILIO_ACCOUNT_SID = "AC6121cafe191c4679edba98f34c79c237"
    const val TWILIO_AUTH_TOKEN = "fc11cb4b70009b3b61e144c06a628392"
    const val TWILIO_FROM_NUMBER = "+18103775029"
    const val EMERGENCY_CONTACT_NUMBER = "+919778176499"

    // Rich Media (ImgBB)
    const val IMGBB_API_KEY = "4f832985f377cdfc85761ae927997a96"

    // WhatsApp Alert
    const val ENABLE_WHATSAPP = true
    val WHATSAPP_TO_NUMBERS = listOf(
        "whatsapp:+919778176499",
        "whatsapp:+918921365016"
    )
    const val WHATSAPP_FROM_NUMBER = "whatsapp:+14155238886"

    // Alarm Settings
    const val ALARM_BEEP_COUNT = 5
    const val ALARM_BEEP_MS = 800

    // Night Mode Settings
    const val NIGHT_BRIGHTNESS_THRESHOLD = 60
    const val NIGHT_ALPHA = 1.8f
    const val NIGHT_BETA = 40f
}
