package com.guardia.fall_detector

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import okhttp3.Call
import okhttp3.Callback
import okhttp3.Credentials
import okhttp3.FormBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object AlertManager {
    private val client = OkHttpClient()

    /**
     * Replicates `escalation_speak` and `send_external_alert` from alerts.py.
     * Uploads the bitmap snapshot to ImgBB, then uses the URL to send SMS/WhatsApp via Twilio.
     */
    fun triggerEscalationAlert(bitmap: Bitmap?) {
        val timestamp = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        val baseMessage = "\uD83D\uDEA8 GUARDIA EMERGENCY ALERT\n---------------------------\nA fall was confirmed at $timestamp.\nThe user did not respond to local alerts."

        if (bitmap != null) {
            uploadImageToImgBB(bitmap) { mediaUrl ->
                val finalMsg = if (mediaUrl != null) {
                    "$baseMessage\n\n\uD83D\uDCF8 View Capture: $mediaUrl\n\nPlease check on them immediately."
                } else {
                    "$baseMessage\n(Image upload failed or unavailable)\n\nPlease check on them immediately."
                }
                sendTwilioAlerts(finalMsg, mediaUrl)
            }
        } else {
            sendTwilioAlerts("$baseMessage\n(Camera snapshot unavailable)\n\nPlease check on them immediately.", null)
        }
    }

    private fun uploadImageToImgBB(bitmap: Bitmap, onResult: (String?) -> Unit) {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
        val byteArray = stream.toByteArray()
        val base64Image = Base64.encodeToString(byteArray, Base64.NO_WRAP)

        val formBody = FormBody.Builder()
            .add("key", Config.IMGBB_API_KEY)
            .add("image", base64Image)
            .build()

        val request = Request.Builder()
            .url("https://api.imgbb.com/1/upload")
            .post(formBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("AlertManager", "ImgBB Upload failed", e)
                onResult(null)
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val resBody = response.body?.string()
                    val json = JSONObject(resBody ?: "")
                    val url = json.getJSONObject("data").getString("url")
                    onResult(url)
                } catch (e: Exception) {
                    Log.e("AlertManager", "ImgBB Parse failed", e)
                    onResult(null)
                }
            }
        })
    }

    private fun sendTwilioAlerts(message: String, mediaUrl: String?) {
        val twilioUrl = "https://api.twilio.com/2010-04-01/Accounts/${Config.TWILIO_ACCOUNT_SID}/Messages.json"
        val credential = Credentials.basic(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

        fun dispatch(from: String, to: String) {
            val bodyBuilder = FormBody.Builder()
                .add("To", to)
                .add("From", from)
                .add("Body", message)
            if (mediaUrl != null) {
                bodyBuilder.add("MediaUrl", mediaUrl)
            }

            val request = Request.Builder()
                .url(twilioUrl)
                .header("Authorization", credential)
                .post(bodyBuilder.build())
                .build()

            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    Log.e("AlertManager", "Twilio sending failed to $to", e)
                }

                override fun onResponse(call: Call, response: Response) {
                    if (response.isSuccessful) {
                        Log.d("AlertManager", "Twilio alert sent successfully to $to")
                    } else {
                        Log.e("AlertManager", "Twilio alert error: ${response.code} ${response.body?.string()}")
                    }
                }
            })
        }

        if (Config.ENABLE_SMS) {
            dispatch(Config.TWILIO_FROM_NUMBER, Config.EMERGENCY_CONTACT_NUMBER)
        }
        if (Config.ENABLE_WHATSAPP) {
            for (number in Config.WHATSAPP_TO_NUMBERS) {
                dispatch(Config.WHATSAPP_FROM_NUMBER, number)
            }
        }
    }
}
