// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorFilter
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.drawable.Drawable

class ItemDrawable(rect: Rect, lbl: String, scre: String, iColor:Int = Color.YELLOW) : Drawable() {
    private val boundingRectPaint = Paint().apply {
        style = Paint.Style.STROKE
        color = iColor
        strokeWidth = 5F
        alpha = 200
    }

    private val contentRectPaint = Paint().apply {
        style = Paint.Style.FILL
        color = iColor
        alpha = 255
    }

    private val contentTextPaint = Paint().apply {
        color = Color.DKGRAY
        alpha = 255
        textSize = 36F
    }

    private val boundingRect = rect
    private val label = lbl
    private val score = scre
    private val contentPadding = 25
    private var textWidth = contentTextPaint.measureText(label + " " + score).toInt()
    private val textRectHeight = contentPadding*2 - contentPadding/4

    override fun draw(canvas: Canvas) {
        canvas.drawRect(boundingRect, boundingRectPaint)
        val left = if(boundingRect.left < 0) 0 else boundingRect.left
        val top = if(boundingRect.top - textRectHeight < 0) 0 else boundingRect.top
        canvas.drawRect(
            Rect(
                left, if(top == 0) contentPadding/4 else top - contentPadding/4,
                left + textWidth + contentPadding*2, if(top == 0) contentPadding*2 else top - contentPadding*2),
            contentRectPaint
        )
        canvas.drawText(
            label + " " + score, (left + contentPadding/4).toFloat(),
            (if(top == 0) contentPadding*1.5 else top - contentPadding/2).toFloat(),
            contentTextPaint
        )
    }

    fun getBoundingRect(): Rect{
        return boundingRect
    }

    fun getLabel(): String{
        return label
    }

    fun getScore(): String{
        return score
    }

    override fun setAlpha(alpha: Int) {
        boundingRectPaint.alpha = alpha
        contentRectPaint.alpha = alpha
        contentTextPaint.alpha = alpha
    }

    override fun setColorFilter(colorFiter: ColorFilter?) {
        boundingRectPaint.colorFilter = colorFilter
        contentRectPaint.colorFilter = colorFilter
        contentTextPaint.colorFilter = colorFilter
    }

    @Deprecated("Deprecated in Java")
    override fun getOpacity(): Int {
        return PixelFormat.TRANSLUCENT
    }
}
