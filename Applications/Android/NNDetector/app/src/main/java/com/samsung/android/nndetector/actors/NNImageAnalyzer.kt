// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import kotlin.concurrent.thread
import kotlin.math.roundToInt
import kotlin.properties.Delegates

class NNImageAnalyzer(detModelPointer: Long, recModelPointer: Long,
                      detFolder: String, recFolder: String,
                      dispMaxHeight:Int = 2320, private val listener: (results:Array<Result>) -> Unit) : ImageAnalysis.Analyzer {
    companion object {
        private const val TAG = "NNImageAnalyzer"
        private const val widthCalib = -0.19
        private const val heightCalib = -0.008
        private const val heightWidthRatio = 48.0/64.0 // widht/height of image input to nntrainer
    }

    private lateinit var mDetFolder: String
    private lateinit var mRecFolder: String
    private var mDetModelPointer by Delegates.notNull<Long>()
    private var mRecModelPointer by Delegates.notNull<Long>()

    private var previewScreenHeight by Delegates.notNull<Int>()
    private var previewScreenWidth by Delegates.notNull<Int>()

    private lateinit var tester: NNImageTester

    init{
        mDetFolder = detFolder
        mRecFolder = recFolder

        mDetModelPointer = detModelPointer
        mRecModelPointer = recModelPointer

        initializeInferenceDir()

        previewScreenHeight = dispMaxHeight
        previewScreenWidth = (previewScreenHeight*heightWidthRatio).toInt()

        tester = NNImageTester(mRecFolder, mDetModelPointer, mRecModelPointer, null)
    }


    external fun runDetector(modelPath: Array<String>, modelPointer: Long) : String

    protected fun finalize(){
        initializeInferenceDir()
    }

    override fun analyze(image: ImageProxy) {
        val bmp = imageProxyToBitmap(image).rotate(90F)
        val jpgPath: String = createJpegFile(bmp, mDetFolder+"/Test", "test")
        Log.i(TAG, "JPG: " + jpgPath)

        Log.i(TAG, "starting to run detection $mDetFolder")
        val args = arrayOf<String>(
            "NNDetector",
            mDetFolder,
            DataShared.detInputImgDim.toString(),
            DataShared.detOutDim.toString(),
            DataShared.detAnchorNum.toString()
        )

        thread(start=true) {
            val ret: String = runDetector(args, mDetModelPointer)
            Log.i(TAG, "NDK result: " + ret)

            val dirDet = File(mDetFolder + "/Test")
            for (file in dirDet.listFiles()!!) {
                file.delete()
            }

            val strList: List<String> = ret.split(",")
            val count: Int = strList[0].toInt()
            var results: Array<Result> = arrayOf()
            if (count > 0) {
                for (i in 1 until strList.size) {
                    val posInfo: List<String> = strList[i].split("/")
                    val pos: List<String> = posInfo[0].split(" ")
                    val result: Result = Result()

                    if(DataShared.isTrained && DataShared.ttMap.containsValue(posInfo[1])) {
                        result.label = getPersonalLabel(bmp, pos, "Test" + i)

                        val dirRec = File(mRecFolder+"/Test")
                        for(file in dirRec.listFiles()!!){
                            file.delete()
                        }
                    }
                    else
                        result.label = posInfo[1]

                    val left: Int = ((pos[0].toFloat() + widthCalib) * previewScreenWidth).toInt()
                    val top: Int = ((pos[1].toFloat() + heightCalib) * previewScreenHeight).toInt()
                    val right: Int = ((pos[2].toFloat() + widthCalib) * previewScreenWidth).toInt()
                    val bottom: Int =
                        ((pos[3].toFloat() + heightCalib) * previewScreenHeight).toInt()

                    Log.i(TAG, "bounding info: " + left + " " + top + " " + right + " " + bottom)
                    result.rect.set(left, top, right, bottom)

                    if (posInfo[2].toFloat() < 0.5f) {
                        result.label = "unknown"
                    }
                    result.score =
                        (((posInfo[2].toFloat()) * 100.0).roundToInt() / 100.0).toString()

                    results += result
                }
            } else {
                Log.i(TAG, "result is empty")
            }
            listener(results)
            image.close()
        }
    }


    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuf = image.planes[0].buffer
        val uBuf = image.planes[1].buffer
        val vBuf = image.planes[2].buffer

        val ySize = yBuf.remaining()
        val uSize = uBuf.remaining()
        val vSize = vBuf.remaining()

        val byteArray = ByteArray(ySize + uSize + vSize)

        yBuf.get(byteArray, 0, ySize)
        vBuf.get(byteArray, ySize, vSize)
        uBuf.get(byteArray, ySize + vSize, uSize)

        val yuvImage = YuvImage(byteArray, ImageFormat.NV21, image.width, image.height, null)
        val byteArrayOutputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, byteArrayOutputStream)
        val imageByteArray = byteArrayOutputStream.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageByteArray, 0, imageByteArray.size)

        return bitmap
    }

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    private fun getPersonalLabel(bitmap: Bitmap, boundingPoint: List<String>, bmpName: String): String {
        val left: Int = (boundingPoint[0].toFloat() * 480).toInt()
        val top: Int = (boundingPoint[1].toFloat() * 640).toInt()
        val right: Int = (boundingPoint[2].toFloat() * 480).toInt()
        val bottom: Int = (boundingPoint[3].toFloat() * 640).toInt()

        val croppedBitmap = bitmap.let {
            Bitmap.createBitmap(
                it, left, top, (right.minus(left)), (bottom.minus(top))
            )
        }

        croppedBitmap?.let {
            createJpegFile(it, mRecFolder+"/Test", bmpName)
        }
        return tester.instantTest()
    }

    @SuppressLint("SimpleDateFormat")
    @Throws(IOException::class)
    private fun createJpegFile(bitmap: Bitmap, folderPath: String, bmpName: String): String{
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val dir: File? = File(folderPath)
        val file = File.createTempFile("JPG_${timeStamp}_${bmpName}_", ".jpg", dir)

        if (file.exists())
            file.delete()

        try {
            val out = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            out.flush()
            out.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return file.absolutePath
    }

    private fun initializeInferenceDir(){
        val detDir = File(mDetFolder+"/Test")
        for(file in detDir.listFiles()!!){
            file.delete()
        }

        val recDir = File(mRecFolder+"/Test")
        for(file in recDir.listFiles()!!){
            file.delete()
        }
    }

    public class Result{
        var label: String
        var score: String
        var rect: Rect

        init{
            label = ""
            score = ""
            rect = Rect()
        }
    }
}
