package com.guardia.fall_detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import kotlin.math.min
import kotlin.math.max

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var poses: List<Pose> = emptyList()
    private var states: Map<Int, PersonState> = emptyMap()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    
    private var fps: Float = 0f
    private var isNightMode: Boolean = false
    private var alertTriggered: Boolean = false
    private var alertTime: Long = 0L
    private var escalated: Boolean = false
    private var currentTime: Long = System.currentTimeMillis()
    private var isFrontCamera: Boolean = false

    private val density = context.resources.displayMetrics.density

    // Paints
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        strokeJoin = Paint.Join.ROUND
    }
    private val skeletonPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val jointPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.FILL
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 14f * density
        typeface = Typeface.DEFAULT_BOLD
    }
    private val panelBgPaint = Paint().apply {
        color = Color.argb(190, 15, 15, 15)
        style = Paint.Style.FILL
    }
    private val accentLinePaint = Paint().apply {
        color = Color.rgb(0, 200, 255) // Cyan/Blue accent from python
        strokeWidth = 1f * density
    }

    // Diagnostic border to confirm view visibility requested by user
    private val debugBorderPaint = Paint().apply {
        color = Color.MAGENTA
        style = Paint.Style.STROKE
        strokeWidth = 2f
        alpha = 100
    }
    
    // Body logic (YOLOv8)
    private val bodyConnections = listOf(
        Pair(5, 6), Pair(5, 7), Pair(7, 9), Pair(6, 8), Pair(8, 10), 
        Pair(5, 11), Pair(6, 12), Pair(11, 12),                    
        Pair(11, 13), Pair(13, 15), Pair(12, 14), Pair(14, 16)      
    )

    fun updateData(
        poses: List<Pose>, states: Map<Int, PersonState>, imgWidth: Int, imgHeight: Int,
        fps: Float, isNightMode: Boolean, alertTriggered: Boolean, alertTime: Long,
        escalated: Boolean, currentTime: Long, isFrontCamera: Boolean = true
    ) {
        this.poses = poses
        this.states = states
        this.imageWidth = imgWidth
        this.imageHeight = imgHeight
        this.fps = fps
        this.isNightMode = isNightMode
        this.alertTriggered = alertTriggered
        this.alertTime = alertTime
        this.escalated = escalated
        this.currentTime = currentTime
        this.isFrontCamera = isFrontCamera
        invalidate() // Using invalidate() to force a redraw on the main thread
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Draw diagnostic border around the entire overlay
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), debugBorderPaint)

        if (imageWidth <= 0 || imageHeight <= 0) return

        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        // 1. Draw Poses & Skeletons
        canvas.save()
        if (isFrontCamera) {
            canvas.scale(-1f, 1f, width / 2f, height / 2f)
        }

        for (pose in poses) {
            val state = states[pose.id]
            val status = state?.status ?: "normal"
            val color = when (status) {
                "emergency", "inactive" -> Color.RED
                "fallen_waiting" -> Color.rgb(255, 165, 0)
                else -> Color.GREEN
            }
            boxPaint.color = color

            val left = pose.boundingBox.left * scaleX
            val top = pose.boundingBox.top * scaleY
            val right = pose.boundingBox.right * scaleX
            val bottom = pose.boundingBox.bottom * scaleY

            canvas.drawRect(left, top, right, bottom, boxPaint)

            for (connection in bodyConnections) {
                val s = pose.keypoints.getOrNull(connection.first)
                val e = pose.keypoints.getOrNull(connection.second)
                if (s != null && e != null && s.score > 0.05f && e.score > 0.05f) {
                    canvas.drawLine(s.x * scaleX, s.y * scaleY, e.x * scaleX, e.y * scaleY, skeletonPaint)
                }
            }
            for (kpt in pose.keypoints) {
                if (kpt.score > 0.05f) {
                    canvas.drawCircle(kpt.x * scaleX, kpt.y * scaleY, 6f, jointPaint)
                }
            }
        }
        canvas.restore()

        // 2. HUD Drawing (authentic python replication)
        val hudW = 230f * density
        canvas.drawRect(0f, 0f, hudW, height.toFloat(), panelBgPaint)
        canvas.drawLine(hudW, 0f, hudW, height.toFloat(), accentLinePaint)

        var y = 30f * density
        
        textPaint.color = Color.rgb(0, 200, 255)
        textPaint.textSize = 22f * density
        canvas.drawText("GUARDIA", 12f * density, y, textPaint)
        
        y += 16f * density
        canvas.drawLine(8f * density, y, hudW - 8f * density, y, accentLinePaint)
        y += 24f * density

        // System Status Badge
        var isEmergency = false
        var isWaiting = false
        for (s in states.values) {
            if (s.status == "emergency" || s.status == "inactive") isEmergency = true
            if (s.status == "fallen_waiting") isWaiting = true
        }

        val statusTxt = if (alertTriggered || escalated || isEmergency) "EMERGENCY" 
                        else if (isWaiting) "POSSIBLE FALL" else "MONITORING"
        val statusCol = if (alertTriggered || escalated || isEmergency) Color.RED 
                        else if (isWaiting) Color.rgb(255, 165, 0) else Color.GREEN

        textPaint.textSize = 12f * density
        textPaint.color = Color.rgb(0, 200, 255)
        canvas.drawText("STATUS", 12f * density, y, textPaint)
        
        y += 22f * density
        textPaint.textSize = 20f * density
        textPaint.color = statusCol
        canvas.drawText(statusTxt, 12f * density, y, textPaint)
        
        y += 24f * density
        val grayPaint = Paint(accentLinePaint).apply { color = Color.DKGRAY }
        canvas.drawLine(8f * density, y, hudW - 8f * density, y, grayPaint)
        
        y += 24f * density
        textPaint.textSize = 14f * density
        textPaint.color = Color.WHITE
        canvas.drawText("TRACKING  ${poses.size} person(s)", 12f * density, y, textPaint)
        y += 24f * density

        // Per-person signal breakdown
        for ((id, state) in states) {
            val pCol = when (state.status) {
                "emergency", "inactive" -> Color.RED
                "fallen_waiting" -> Color.rgb(255, 165, 0)
                else -> Color.WHITE
            }
            textPaint.textSize = 14f * density
            textPaint.color = pCol
            canvas.drawText("ID $id — ${state.status.uppercase()}", 12f * density, y, textPaint)
            y += 24f * density

            // Signals VEL SPL A/R
            val signals = listOf(
                Pair("VEL", state.isFallingFast),
                Pair("SPL", state.isSpineTilted),
                Pair("A/R", state.isHorizontal)
            )
            var sx = 14f * density
            for ((label, active) in signals) {
                val boxCol = if (active) Color.RED else Color.DKGRAY
                val bgp = Paint().apply { color = boxCol; style = Paint.Style.FILL }
                val strokep = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 1f }
                
                canvas.drawRect(sx, y - 14f * density, sx + 32f * density, y + 4f * density, bgp)
                canvas.drawRect(sx, y - 14f * density, sx + 32f * density, y + 4f * density, strokep)
                
                textPaint.textSize = 10f * density
                textPaint.color = Color.WHITE
                canvas.drawText(label, sx + 4f * density, y, textPaint)
                sx += 38f * density
            }
            y += 24f * density

            // Recovery timer
            if (state.status == "fallen_waiting" && state.fallTimestamp != null) {
                val elapsed = currentTime - state.fallTimestamp
                val ratio = min(elapsed.toFloat() / Config.RECOVERY_TIMEOUT_MS, 1.0f)
                
                textPaint.color = Color.rgb(255, 165, 0)
                canvas.drawText("RECOVERY TIMER", 12f * density, y, textPaint)
                y += 12f * density
                
                val pw = hudW - 24f * density
                val px = 12f * density
                canvas.drawRect(px, y, px + pw, y + 8f * density, Paint().apply { color = Color.DKGRAY })
                canvas.drawRect(px, y, px + pw * ratio, y + 8f * density, Paint().apply { color = Color.rgb(255,165,0) })
                canvas.drawRect(px, y, px + pw, y + 8f * density, Paint().apply { color = Color.WHITE; style=Paint.Style.STROKE })
                y += 24f * density
            }
            y += 8f * density
        }

        // 3. Status Bar (Bottom)
        val barY = height - 30f * density
        canvas.drawRect(0f, barY, width.toFloat(), height.toFloat(), panelBgPaint)
        canvas.drawLine(0f, barY, width.toFloat(), barY, accentLinePaint)

        textPaint.textSize = 14f * density
        textPaint.color = Color.WHITE
        canvas.drawText(String.format("FPS %.1f", fps), 12f * density, height - 10f * density, textPaint)

        textPaint.color = if (isNightMode) Color.rgb(255, 165, 0) else Color.GREEN
        val modeTxt = if (isNightMode) "NIGHT MODE ON" else "NORMAL LIGHT"
        canvas.drawText(modeTxt, 100f * density, height - 10f * density, textPaint)

        // 4. Emergency Banner in Center
        if (alertTriggered || escalated || isEmergency) {
            val bw = 320f * density
            val bh = 140f * density
            val bx = (width - bw) / 2f
            val by = (height - bh) / 2f - 40f * density

            val pulse = ((currentTime / 500) % 2) == 0L
            val borderCol = if (pulse) Color.RED else Color.rgb(255, 165, 0)

            canvas.drawRect(bx, by, bx + bw, by + bh, Paint().apply { color = Color.argb(200,0,0,0) })
            canvas.drawRect(bx, by, bx + bw, by + bh, Paint().apply { color = borderCol; style=Paint.Style.STROKE; strokeWidth=6f })

            textPaint.color = Color.RED
            textPaint.textSize = 24f * density
            val emTxt = "! EMERGENCY DETECTED !"
            val tw = textPaint.measureText(emTxt)
            canvas.drawText(emTxt, bx + (bw - tw)/2f, by + 40f * density, textPaint)

            textPaint.color = Color.WHITE
            textPaint.textSize = 18f * density
            val okTxt = "Tap to confirm I AM OKAY"
            val tw2 = textPaint.measureText(okTxt)
            canvas.drawText(okTxt, bx + (bw - tw2)/2f, by + 80f * density, textPaint)

            if (alertTime > 0L && alertTriggered && !escalated) {
                val elapsed = currentTime - alertTime
                val ratio = max(0f, 1f - elapsed.toFloat() / Config.ALERT_RESPONSE_TIMEOUT_MS)
                val barCol = if (ratio > 0.5f) Color.GREEN else (if (ratio > 0.2f) Color.rgb(255,165,0) else Color.RED)
                
                val pw = bw - 40f * density
                val px = bx + 20f * density
                val py = by + 100f * density
                canvas.drawRect(px, py, px + pw, py + 10f * density, Paint().apply { color=Color.DKGRAY })
                canvas.drawRect(px, py, px + pw * ratio, py + 10f * density, Paint().apply { color=barCol })
                
                val secsLeft = max(0L, (Config.ALERT_RESPONSE_TIMEOUT_MS - elapsed) / 1000 + 1)
                textPaint.textSize = 12f * density
                textPaint.color = Color.LTGRAY
                canvas.drawText("Auto-escalate in ${secsLeft}s", px, py + 24f * density, textPaint)
            }
        }
    }
}
