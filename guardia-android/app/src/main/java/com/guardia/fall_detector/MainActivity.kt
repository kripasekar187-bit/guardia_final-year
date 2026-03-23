package com.guardia.fall_detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var cameraManager: CameraManager
    private lateinit var poseEstimator: PoseEstimator
    private lateinit var fallDetector: FallDetectorLogic
    private lateinit var overlayView: OverlayView
    private lateinit var tts: TextToSpeech

    private var alertTriggered = false
    private var escalated = false
    private var alertTime = 0L
    
    // FPS tracking
    private var fpsCounter = 0
    private var fpsDisplay = 0f
    private var fpsTimer = System.currentTimeMillis()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        overlayView = findViewById(R.id.overlayView)

        // Force overlay to front
        overlayView.bringToFront()

        poseEstimator = PoseEstimator(this)
        fallDetector = FallDetectorLogic()
        tts = TextToSpeech(this, this)

        checkPermissionsAndStart()
    }

    private fun checkPermissionsAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val previewView: PreviewView = findViewById(R.id.previewView)
        
        cameraManager = CameraManager(this, this, previewView) { imageProxy ->
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            var bitmap = imageProxy.toBitmap()

            // Fix for "skeletal frame is not working": CameraX sensor is often 90 degrees offset.
            // We must manually rotate the bitmap upright so YOLO sees upright people, 
            // mapping exactly to the portrait screen OverlayView dimensions!
            if (rotationDegrees != 0) {
                val matrix = android.graphics.Matrix()
                matrix.postRotate(rotationDegrees.toFloat())
                bitmap = android.graphics.Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
            }
            
            val isNight = NightModeManager.isNightMode(bitmap)

            // Run inference
            val poses = poseEstimator.estimatePoses(bitmap)
            
            // Analyze poses for falls
            val currentTime = System.currentTimeMillis()
            val states = fallDetector.analyzePoses(poses, currentTime)

            var isEmergency = false
            var isWaiting = false
            for (state in states.values) {
                if (state.status == "emergency" || state.status == "inactive") isEmergency = true
                if (state.status == "fallen_waiting") isWaiting = true
            }

             if (isEmergency && !alertTriggered && !escalated) {
                 alertTriggered = true
                 alertTime = currentTime
                 tts.speak("Emergency Detected. Are you okay?", TextToSpeech.QUEUE_FLUSH, null, null)
             }

             if ((alertTriggered || escalated) && !isEmergency && !isWaiting) {
                 AudioAlarm.stop()
                 alertTriggered = false
                 escalated = false
                 tts.speak("Okay, alert cancelled.", TextToSpeech.QUEUE_FLUSH, null, null)
             }

             if (alertTriggered && !escalated) {
                 if (currentTime - alertTime > Config.ALERT_RESPONSE_TIMEOUT_MS) {
                     escalated = true
                     alertTriggered = false
                     tts.speak("Emergency verified. Escalating.", TextToSpeech.QUEUE_FLUSH, null, null)
                     AudioAlarm.play()
                     AlertManager.triggerEscalationAlert(bitmap)
                 }
             }

            // FPS calculation
            fpsCounter++
            if (currentTime - fpsTimer >= 1000) {
                fpsDisplay = fpsCounter / ((currentTime - fpsTimer) / 1000f)
                fpsCounter = 0
                fpsTimer = currentTime
            }

            // Update UI entirely via OverlayView's Canvas API mirroring OpenCV
            runOnUiThread {
                overlayView.updateData(
                    poses = poses,
                    states = states,
                    imgWidth = bitmap.width,
                    imgHeight = bitmap.height,
                    fps = fpsDisplay,
                    isNightMode = isNight,
                    alertTriggered = alertTriggered,
                    alertTime = alertTime,
                    escalated = escalated,
                    currentTime = currentTime,
                    isFrontCamera = true
                )
            }

            imageProxy.close()
        }
        
        cameraManager.startCamera()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.stopCamera()
        tts.stop()
        tts.shutdown()
    }
}
