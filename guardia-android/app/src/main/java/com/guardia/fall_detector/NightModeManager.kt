package com.guardia.fall_detector

import android.graphics.Bitmap
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter

object NightModeManager {

    /**
     * Night mode is currently disabled as requested.
     */
    fun isNightMode(bitmap: Bitmap): Boolean {
        return false
    }

    /**
     * Returns a neutral filter since night mode is disabled.
     */
    fun getNightFilter(): ColorMatrixColorFilter {
        val matrix = ColorMatrix()
        return ColorMatrixColorFilter(matrix)
    }
}
