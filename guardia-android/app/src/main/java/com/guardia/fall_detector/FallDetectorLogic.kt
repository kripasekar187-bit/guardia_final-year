package com.guardia.fall_detector

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

data class PersonState(
    var status: String = "normal",
    var isFallingFast: Boolean = false,
    var isSpineTilted: Boolean = false,
    var isHorizontal: Boolean = false,
    var fallTimestamp: Long? = null,
    var velocityEma: Float? = null,
    
    // Additional tracking internal variables mirroring Python logic
    var prevTorsoY: Float? = null,
    var prevTime: Long? = null,
    var lastActiveTime: Long = 0L,
    var handRaisedFrames: Int = 0,
    var recoveryStartTime: Long? = null,
    var lastSeen: Long = 0L
)

class FallDetectorLogic {

    companion object {
        const val FALLING_VELOCITY_THRESHOLD = 0.12f 
        const val ASPECT_RATIO_HORIZONTAL = 0.84f
        const val VELOCITY_EMA_ALPHA = 0.35f
        const val MIN_KP_CONF = 0.30f
        const val SPINE_ANGLE_THRESHOLD_DEG = 45.0f
        const val ACTIVITY_MOTION_THRESHOLD = 0.03f
        
        const val RECOVERY_TIMEOUT_SECONDS = 3.5f
        const val RECOVERY_LATCH_SECONDS = 3.5f
        const val INACTIVITY_THRESHOLD = 2.0f
        const val STALE_TRACK_TIMEOUT = 5.0f
        const val GESTURE_CONFIRM_FRAMES = 5
    }

    private val personStates = mutableMapOf<Int, PersonState>()

    fun analyzePoses(currentPoses: List<Pose>, currentTimeMs: Long): Map<Int, PersonState> {

        for (pose in currentPoses) {
            val state = personStates.getOrPut(pose.id) {
                PersonState(lastSeen = currentTimeMs, lastActiveTime = currentTimeMs, prevTime = currentTimeMs)
            }
            state.lastSeen = currentTimeMs

            val boxW = pose.boundingBox.width
            val boxH = pose.boundingBox.height
            if (boxH <= 0) continue

            fun kp(idx: Int): Keypoint? {
                val kpt = pose.keypoints.getOrNull(idx)
                return if (kpt != null && kpt.score >= MIN_KP_CONF) kpt else null
            }

            // 1. Aspect Ratio
            val aspectRatio = boxW / boxH
            var isHorizontal = aspectRatio > ASPECT_RATIO_HORIZONTAL

            // 2. Spine Angle
            var spineAngleDeg: Float? = null
            var isSpineTilted = false

            val lShoulder = kp(5)
            val rShoulder = kp(6)
            val lHip = kp(11)
            val rHip = kp(12)

            var shY: Float? = null
            var hpY: Float? = null

            if (lShoulder != null && rShoulder != null && lHip != null && rHip != null) {
                val shX = (lShoulder.x + rShoulder.x) / 2f
                shY = (lShoulder.y + rShoulder.y) / 2f
                val hpX = (lHip.x + rHip.x) / 2f
                hpY = (lHip.y + rHip.y) / 2f

                val dx = hpX - shX
                val dy = hpY - shY
                val spineLen = hypot(dx, dy)

                if (spineLen > 5.0f) {
                    spineAngleDeg = Math.toDegrees(atan2(abs(dx).toDouble(), abs(dy).toDouble())).toFloat()
                    isSpineTilted = spineAngleDeg > SPINE_ANGLE_THRESHOLD_DEG

                    val spineNorm = (hpY - shY) / boxH
                    if (spineNorm > 0.30f) {
                        isHorizontal = false
                    }
                }
            }

            // 3. Centroid Velocity (EMA)
            var isFallingFast = false
            val torsoY = if (hpY != null && shY != null) {
                (hpY + shY) / 2f
            } else {
                (pose.boundingBox.top + pose.boundingBox.bottom) / 2f
            }

            if (state.prevTorsoY != null && state.prevTime != null) {
                val dtSec = (currentTimeMs - state.prevTime!!) / 1000f
                if (dtSec > 0f) {
                    val rawDelta = (torsoY - state.prevTorsoY!!) / boxH
                    val rawVelocity = rawDelta / dtSec

                    val prevEma = state.velocityEma
                    val emaVelocity = if (prevEma == null) {
                        rawVelocity
                    } else {
                        (VELOCITY_EMA_ALPHA * rawVelocity) + ((1f - VELOCITY_EMA_ALPHA) * prevEma)
                    }
                    state.velocityEma = emaVelocity

                    isFallingFast = emaVelocity > FALLING_VELOCITY_THRESHOLD

                    if (abs(rawDelta) > ACTIVITY_MOTION_THRESHOLD && !isHorizontal) {
                        state.lastActiveTime = currentTimeMs
                    }
                }
            }
            state.prevTorsoY = torsoY
            state.prevTime = currentTimeMs

            // 4. Crouch Checking
            var isCrouching = false
            val lAnkle = kp(15)
            val rAnkle = kp(16)

            if (lHip != null && rHip != null && (lAnkle != null || rAnkle != null)) {
                val hipYCombined = (lHip.y + rHip.y) / 2f
                var ankleYCombined = 0f
                var count = 0
                if (lAnkle != null) { ankleYCombined += lAnkle.y; count++ }
                if (rAnkle != null) { ankleYCombined += rAnkle.y; count++ }
                ankleYCombined /= count

                val normalizedGap = (ankleYCombined - hipYCombined) / boxH
                isCrouching = normalizedGap < 0.18f
            }

            // Camera proximity
            val lKnee = kp(13); val rKnee = kp(14)
            val lowerLimbsVisible = (lKnee != null || rKnee != null || lAnkle != null || rAnkle != null)
            if (!lowerLimbsVisible && isHorizontal) {
                isHorizontal = false
            }

            // 5. Hand Raise Gesture
            var isHandRaised = false
            val nose = kp(0)
            val lWrist = kp(9)
            val rWrist = kp(10)

            if (nose != null) {
                if (lWrist != null && lWrist.y < nose.y) isHandRaised = true
                if (rWrist != null && rWrist.y < nose.y) isHandRaised = true
            }

            if (isHandRaised) {
                state.handRaisedFrames++
            } else {
                state.handRaisedFrames = 0
            }

            val isHandRaisedConfirmed = state.handRaisedFrames >= GESTURE_CONFIRM_FRAMES

            // 6. Fall Decision Logic
            var primaryCount = 0
            if (isFallingFast) primaryCount++
            if (isSpineTilted) primaryCount++
            if (isHorizontal) primaryCount++

            val suppressed = isCrouching || isHandRaisedConfirmed
            var activeFall = (primaryCount >= 2) && !suppressed

            if ((state.status == "fallen_waiting" || state.status == "emergency") && isHorizontal && !isHandRaisedConfirmed) {
                activeFall = true
            }

            if (activeFall) {
                state.recoveryStartTime = null

                if (state.status == "normal") {
                    state.status = "fallen_waiting"
                    state.fallTimestamp = currentTimeMs
                } else if (state.status == "fallen_waiting") {
                    val elapsedSec = (currentTimeMs - (state.fallTimestamp ?: currentTimeMs)) / 1000f
                    if (elapsedSec >= RECOVERY_TIMEOUT_SECONDS) {
                        state.status = "emergency"
                    }
                }
            } else {
                if (state.status == "fallen_waiting" || state.status == "emergency" || state.status == "inactive") {
                    if (isHandRaisedConfirmed) {
                        state.status = "normal"
                        state.fallTimestamp = null
                    } else {
                        if (state.recoveryStartTime == null) {
                            state.recoveryStartTime = currentTimeMs
                        }
                        val elapsedRecovery = (currentTimeMs - state.recoveryStartTime!!) / 1000f
                        if (elapsedRecovery >= RECOVERY_LATCH_SECONDS) {
                            state.status = "normal"
                            state.fallTimestamp = null
                            state.recoveryStartTime = null
                        }
                    }
                } else {
                    state.recoveryStartTime = null
                }
            }

            // 7. Inactivity failsafe
            if (isHorizontal && state.status != "emergency" && state.status != "inactive" && !isHandRaisedConfirmed) {
                val inactiveForSec = (currentTimeMs - state.lastActiveTime) / 1000f
                if (inactiveForSec > INACTIVITY_THRESHOLD) {
                    state.status = "inactive"
                }
            }

            // Debug state mapping out
            state.isFallingFast = isFallingFast
            state.isSpineTilted = isSpineTilted
            state.isHorizontal = isHorizontal
        }

        // Cleanup stale disconnected tracking sessions
        val staleIds = personStates.filter { (currentTimeMs - it.value.lastSeen) / 1000f > STALE_TRACK_TIMEOUT }.keys
        staleIds.forEach { personStates.remove(it) }

        return personStates
    }
}
