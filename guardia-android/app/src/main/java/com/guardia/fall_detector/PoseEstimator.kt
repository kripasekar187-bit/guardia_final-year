package com.guardia.fall_detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class PoseEstimator(context: Context) {
    private var interpreter: Interpreter? = null
    private val modelInputSize = 640
    private val confidenceThreshold = 0.45f
    private val nmsThreshold = 0.45f

    private var nextId = 1
    private val activeTracks = mutableListOf<Pose>()

    init {
        try {
            val assetManager = context.assets
            val fileDescriptor = assetManager.openFd("yolov8n-pose.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d("PoseEstimator", "Model loaded successfully.")
        } catch (e: Exception) {
            Log.e("PoseEstimator", "Error loading model.", e)
        }
    }

    fun estimatePoses(bitmap: Bitmap): List<Pose> {
        val currentInterpreter = interpreter ?: return emptyList()
        val startTime = System.currentTimeMillis()

        // 1. Precise Letterbox padded scaling logic (avoids stretching humans out of recognizable proportions!)
        val scale = min(modelInputSize.toFloat() / bitmap.width, modelInputSize.toFloat() / bitmap.height)
        val newWidth = (bitmap.width * scale).toInt()
        val newHeight = (bitmap.height * scale).toInt()

        val resized = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        val padded = Bitmap.createBitmap(modelInputSize, modelInputSize, Bitmap.Config.ARGB_8888)
        val pc = android.graphics.Canvas(padded)
        pc.drawColor(android.graphics.Color.BLACK)
        
        val padX = (modelInputSize - newWidth) / 2f
        val padY = (modelInputSize - newHeight) / 2f
        pc.drawBitmap(resized, padX, padY, null)

        val tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(padded)
        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0f, 255f))
            .build()
        val processedImage = imageProcessor.process(tensorImage)

        val outputBuffer = Array(1) { Array(56) { FloatArray(8400) } }
        currentInterpreter.run(processedImage.buffer, outputBuffer)

        val detections = mutableListOf<Pose>()
        val output = outputBuffer[0]

        for (i in 0 until 8400) {
            val score = output[4][i]
            if (score > confidenceThreshold) {
                var cx = output[0][i]
                var cy = output[1][i]
                var w = output[2][i]
                var h = output[3][i]

                // Guard for YOLO normalized exports! If outputs are 0..1, they must be scaled by 640
                val isNormalized = (w <= 1.5f && h <= 1.5f)
                if (isNormalized) {
                    cx *= modelInputSize
                    cy *= modelInputSize
                    w *= modelInputSize
                    h *= modelInputSize
                }

                // Unpad coordinates mathematically translating off the letterbox back entirely to the generic raw dimension
                val cx_unpadded = (cx - padX) / scale
                val cy_unpadded = (cy - padY) / scale
                val w_unpadded = w / scale
                val h_unpadded = h / scale

                val left = cx_unpadded - w_unpadded / 2f
                val top = cy_unpadded - h_unpadded / 2f
                val right = cx_unpadded + w_unpadded / 2f
                val bottom = cy_unpadded + h_unpadded / 2f

                val bbox = BoundingBox(left, top, right, bottom)

                val kpts = mutableListOf<Keypoint>()
                for (k in 0 until 17) {
                    var rawKx = output[5 + k * 3][i]
                    var rawKy = output[6 + k * 3][i]
                    if (isNormalized) {
                        rawKx *= modelInputSize
                        rawKy *= modelInputSize
                    }
                    val kx = (rawKx - padX) / scale
                    val ky = (rawKy - padY) / scale
                    val kconf = output[7 + k * 3][i]
                    kpts.add(Keypoint(kx, ky, kconf))
                }

                detections.add(Pose(0, bbox, kpts, startTime))
            }
        }

        val nmsPoses = applyNMS(detections)
        return trackPoses(nmsPoses)
    }

    private fun applyNMS(poses: List<Pose>): List<Pose> {
        val sorted = poses.sortedByDescending { it.boundingBox.width * it.boundingBox.height }
        val result = mutableListOf<Pose>()
        for (pose in sorted) {
            var keep = true
            for (selected in result) {
                if (calculateIoU(pose.boundingBox, selected.boundingBox) > nmsThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) result.add(pose)
        }
        return result
    }

    private fun calculateIoU(b1: BoundingBox, b2: BoundingBox): Float {
        val iLeft = max(b1.left, b2.left)
        val iTop = max(b1.top, b2.top)
        val iRight = min(b1.right, b2.right)
        val iBottom = min(b1.bottom, b2.bottom)
        val iW = max(0f, iRight - iLeft)
        val iH = max(0f, iBottom - iTop)
        val iArea = iW * iH
        val a1 = b1.width * b1.height
        val a2 = b2.width * b2.height
        return iArea / (a1 + a2 - iArea)
    }

    private fun trackPoses(newPoses: List<Pose>): List<Pose> {
        val tracked = mutableListOf<Pose>()
        for (pose in newPoses) {
            var bestMatch: Pose? = null
            var maxIoU = 0.3f 
            for (prev in activeTracks) {
                val iou = calculateIoU(pose.boundingBox, prev.boundingBox)
                if (iou > maxIoU) {
                    maxIoU = iou
                    bestMatch = prev
                }
            }
            val finalPose = if (bestMatch != null) {
                pose.copy(id = bestMatch.id)
            } else {
                pose.copy(id = nextId++)
            }
            tracked.add(finalPose)
        }
        activeTracks.clear()
        activeTracks.addAll(tracked)
        return tracked
    }
}
