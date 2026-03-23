package com.guardia.fall_detector

import android.media.AudioManager
import android.media.ToneGenerator
import kotlin.concurrent.thread

object AudioAlarm {
    @Volatile
    private var isPlaying = false

    /**
     * Replicates the `play_alarm()` function from alerts.py using Android's ToneGenerator
     * to emit a loud siren-like beep for ALARM_BEEP_MS duration, ALARM_BEEP_COUNT times.
     */
    fun play() {
        if (isPlaying) return
        isPlaying = true
        thread {
            // STREAM_ALARM ensures it bypasses normal media volume and rings as an alarm
            val toneGen = ToneGenerator(AudioManager.STREAM_ALARM, 100)
            try {
                for (i in 0 until Config.ALARM_BEEP_COUNT) {
                    if (!isPlaying) break
                    // TONE_SUP_ERROR is a loud high-pitched generic alarm tone similar to 1200hz
                    toneGen.startTone(ToneGenerator.TONE_SUP_ERROR, Config.ALARM_BEEP_MS)
                    // Sleep for the Tone duration + a 150ms gap
                    Thread.sleep(Config.ALARM_BEEP_MS.toLong() + 150)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                toneGen.release()
                isPlaying = false
            }
        }
    }

    /**
     * Stop the alarm loop early if user clicks "I AM OKAY"
     */
    fun stop() {
        isPlaying = false
    }
}
