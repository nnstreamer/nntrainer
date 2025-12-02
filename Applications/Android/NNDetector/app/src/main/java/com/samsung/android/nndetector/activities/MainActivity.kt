// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. HS.Kim <hs0207.kim@samsung.com>
 * 
 * @file   MainActivity.kt
 * @date   24 Oct 2023
 * @brief  Main activity showing object detection
 * @author HS.Kim (hs0207.kim@samsung.com)
 * @bug    No known bugs
 */
package com.samsung.android.nndetector

import android.Manifest
import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView

import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.samsung.android.nndetector.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import kotlin.properties.Delegates

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_PERMISSION_CODE = 1000
        private val PERMISSIONS_NEEDED =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ).toTypedArray()
    }

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var currentFullImagePath: String
    private lateinit var currentCropImagePath: String

    private lateinit var cameraController: LifecycleCameraController

    private lateinit var ctrlBtn: FloatingActionButton
    private lateinit var trainTestCtrlBtn: FloatingActionButton
    private lateinit var resetCtrlBtn: FloatingActionButton
    private lateinit var finCtrlBtn: FloatingActionButton
    private var fabControlOpened by Delegates.notNull<Boolean>()
    private lateinit var needToTrain:String
    private var cntOfNewDataSet by Delegates.notNull<Int>()

    private lateinit var sLock: Any

    external fun createDetectionModel(args: Array<String>): Long
    external fun createRecognitionModel(args: Array<String>): Long


    init {
        System.loadLibrary("simpleshot_jni")
        fabControlOpened = false
        needToTrain = "No"
        cntOfNewDataSet = 0

        sLock = Any()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        supportActionBar?.hide()
        ctrlBtn = viewBinding.fabControl
        ctrlBtn.setOnClickListener {
            toggleControl()
        }

        trainTestCtrlBtn = viewBinding.fabTrainTest
        trainTestCtrlBtn.setOnClickListener {
            cameraController.clearImageAnalysisAnalyzer() // clear analyzer
            synchronized(sLock) { // waiting ending of analizing
                (sLock as Object).wait()
            }

            if(cntOfNewDataSet > 0)
                needToTrain = "Yes"

            val ttIntent = Intent(this, TrainActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("needToTrain", needToTrain)
            }
            startActivity(ttIntent)

            cntOfNewDataSet = 0
            needToTrain = "No"
        }

        resetCtrlBtn = viewBinding.fabReset
        resetCtrlBtn.setOnClickListener{
            cameraController.clearImageAnalysisAnalyzer() // clear analyzer
            synchronized(sLock) { // waiting ending of analizing
                (sLock as Object).wait()
            }

            DataShared.isTrained = false

            DataShared.personalObjectList.clear()
            DataShared.personalObjectList = mutableListOf()
            DataShared.deleteFolder(DataShared.dataFolder+"/data")

            dataReinitize()
            startCamera()
        }

        finCtrlBtn = viewBinding.fabFinish
        finCtrlBtn.setOnClickListener {
            finish()
        }

        nnInitize()

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, PERMISSIONS_NEEDED, REQUEST_PERMISSION_CODE
            )
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun startCamera() {
        val previewView: PreviewView = viewBinding.cameraPreview
        cameraController = LifecycleCameraController(baseContext)

        cameraController.setImageAnalysisAnalyzer(
            ContextCompat.getMainExecutor(this),
            NNImageAnalyzer(DataShared.detModelPointer, DataShared.recModelPointer, DataShared.detFolder,
                                DataShared.recFolder, DataShared.dispMaxHeight) { results ->
                synchronized(sLock) { // notifying ending of analizing
                    (sLock as Object).notify()
                }

                var itemDrawables:Array<ItemDrawable> = arrayOf()

                for(result in results) {
                    val itemDrawable = result.let { ItemDrawable(result.rect, result.label,
                        result.score, Color.CYAN) }
                    itemDrawables += itemDrawable
                }

                previewView.overlay.clear()
                for (itemDrawable in itemDrawables) {
                    previewView.overlay.add(itemDrawable)
                }

                previewView.setOnTouchListener { v: View, e: MotionEvent ->
                    if (e.action == MotionEvent.ACTION_UP) {
                        for (itemDraw in itemDrawables) {
                            val boundingRect: Rect = itemDraw.getBoundingRect()
                            if ((e.x >= boundingRect.left && e.x <= boundingRect.right) &&
                                (e.y >= boundingRect.top && e.y <= boundingRect.bottom)
                            ) {
                                val left = if(boundingRect.left < 0) 0 else boundingRect.left
                                val top = if(boundingRect.top < 0) 0 else boundingRect.top
                                val right = if(boundingRect.right > previewView.bitmap?.width!!) previewView.bitmap?.width else boundingRect.right
                                val bottom = if(boundingRect.bottom > previewView.bitmap?.height!!) previewView.bitmap?.height else boundingRect.bottom

                                currentFullImagePath = "not_used_image" // for future update

                                val croppedBitmap = previewView.bitmap?.let {
                                    Bitmap.createBitmap(
                                        it, left, top, (right?.minus(left)!!), (bottom?.minus(top)!!)
                                    )
                                }

                                currentCropImagePath = croppedBitmap?.let {
                                    createJpegFile(it, true)
                                }.toString()


                                val touchIntent = Intent(this, CropViewActivity::class.java).apply {
                                    Intent.ACTION_VIEW
                                    putExtra("label", if(DataShared.ttMap[itemDraw.getLabel()].isNullOrEmpty()) itemDraw.getLabel() else DataShared.ttMap[itemDraw.getLabel()])
                                    putExtra("score", itemDraw.getScore())
                                    putExtra("cropImagePath", currentCropImagePath)
                                    putExtra("fullImagePath", currentFullImagePath)
                                }
                                startActivity(touchIntent)
                                break
                            }
                        }
                    }
                    false
                }
            }
        )

        cameraController.bindToLifecycle(this)
        previewView.controller = cameraController
    }

    fun dataReinitize(){
        DataShared.trainFolder = DataShared.dataFolder + "/data/train"
        DataShared.makeFolders(DataShared.trainFolder)

        DataShared.testFolder = DataShared.dataFolder + "/data/test"
        DataShared.makeFolders(DataShared.testFolder)

        if(File(DataShared.dataFolder + "/data/data.properties").createNewFile()){
            DataShared.storeFakePropeties("ttMap", "{fake=Empty}")
        }
        DataShared.loadMaps() // initialize map from properties
    }

    fun nnInitize(){
        val programName:String = "NNDetector"
        val detBackbonePath: String = DataShared.detBackbonesFolder + "/yolov8n_float32.tflite"
        val recBackbonePath: String = DataShared.recBackbonesFolder + "/dino_small_patch16_float32.tflite"

        val detArgs =
            arrayOf<String>(programName, detBackbonePath, DataShared.detInputImgDim.toString())
        DataShared.detModelPointer = createDetectionModel(detArgs)
        Log.d(TAG, "create Detection Model Done ${DataShared.detModelPointer}")

        val recArgs =
            arrayOf<String>( programName, recBackbonePath, DataShared.recInputImgDim.toString(),
                DataShared.numClass.toString(), DataShared.knnNormVariant)
        DataShared.recModelPointer = createRecognitionModel(recArgs)
        Log.d(TAG, "create Recognition Model Done ${DataShared.recModelPointer}")

        DataShared.updatePersonalObjectList()
        if(DataShared.personalObjectList.size > 0) {
            val nnImageTrainer = NNImageTrainer(DataShared.trainFolder, DataShared.detModelPointer,
                                                DataShared.recModelPointer, null)
            nnImageTrainer.initialTrain()
        }
    }

    @SuppressLint("SimpleDateFormat")
    @Throws(IOException::class)
    private fun createJpegFile(bitmap: Bitmap, isCropped:Boolean): String {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val dir = File(DataShared.dataFolder+"/data/temp")
        var imagePath:String = ""

        dir.mkdir()
        val file = File.createTempFile((if(isCropped) "" else "full_") + "${timeStamp}_", ".jpg", dir)
            .apply { imagePath = absolutePath }

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
        return imagePath
    }

    fun toggleControl(){
        if(!fabControlOpened) {
            ctrlBtn.setImageResource(R.drawable.controller_extract)
            val test_anim = ObjectAnimator.ofFloat(trainTestCtrlBtn, "translationY", -200f)
            test_anim.duration = 200
            test_anim.start()

            val reset_anim = ObjectAnimator.ofFloat(resetCtrlBtn, "translationY", -400f)
            reset_anim.duration = 200
            reset_anim.start()

            val fin_anim = ObjectAnimator.ofFloat(finCtrlBtn, "translationY", -600f)
            fin_anim.duration = 200
            fin_anim.start()
        } else {
            ctrlBtn.setImageResource(R.drawable.controller_add)
            val test_anim = ObjectAnimator.ofFloat(trainTestCtrlBtn, "translationY", 0f)
            test_anim.duration = 200
            test_anim.start()

            val reset_anim = ObjectAnimator.ofFloat(resetCtrlBtn, "translationY", 0f)
            reset_anim.duration = 200
            reset_anim.start()

            val fin_anim = ObjectAnimator.ofFloat(finCtrlBtn, "translationY", 0f)
            fin_anim.duration = 200
            fin_anim.start()
        }
        fabControlOpened = !fabControlOpened
    }

    private fun allPermissionsGranted() = PERMISSIONS_NEEDED.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        val isNewData = intent?.getStringExtra("newDataset").toString()
        when(isNewData) {
            "Yes" -> {
                cntOfNewDataSet++
                Log.i(TAG, "new dataset exist")
            }

            "No" -> {
                Log.i(TAG, "No new dataset")
            }
        }

        startCamera() // restart camera & analyzer
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSION_CODE) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,"Permissions not granted", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
}
