// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. HS.Kim <hs0207.kim@samsung.com>
 * 
 * @file   TestActivity.kt
 * @date   24 Oct 2023
 * @brief  Test operation with trained personal object
 * @author HS.Kim (hs0207.kim@samsung.com)
 * @bug    No known bugs
 */
package com.samsung.android.nndetector

import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.content.Intent
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.result.ActivityResultCallback
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.samsung.android.nndetector.databinding.ActivityTestBinding
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Timer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.concurrent.timer
import kotlin.properties.Delegates

class TestActivity: AppCompatActivity(){
    private lateinit var binding: ActivityTestBinding
    private lateinit var tester: NNImageTester
    private lateinit var gImageUri: Uri
    private lateinit var userImageDialog: UserImageDialog
    private lateinit var previewView:PreviewView

    private lateinit var ctrlBtn: FloatingActionButton
    private lateinit var galleryBtn: FloatingActionButton
    private lateinit var backCtrlBtn: FloatingActionButton
    private var fabControlOpened by Delegates.notNull<Boolean>()

    private lateinit var blockLayout: LinearLayout
    private lateinit var mWinManager: WindowManager
    private lateinit var mParams: WindowManager.LayoutParams
    private lateinit var isOverLayed: AtomicBoolean

    companion object {
        private const val TAG = "TestActivity"
    }

    init{
        fabControlOpened = false
        isOverLayed = AtomicBoolean(false)
    }

    private val startActivityResult: ActivityResultLauncher<Intent> = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
        ActivityResultCallback {
            if(it.resultCode == RESULT_OK && it.data != null){
                gImageUri = it.data!!.data!!
                Log.i(TAG, gImageUri.toString())
                copyFile(File(getPathFromContentUri(gImageUri)), File(DataShared.testFolder+ "/test"))

                thread(start = true) {
                    tester.test()
                }

                userImageDialog = UserImageDialog(this, gImageUri)
                userImageDialog.show()
                mWinManager.addView(blockLayout, mParams)
            }
        }
    )

    @SuppressLint("RtlHardcoded")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTestBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.hide()

        blockLayout = LinearLayout(this)
        val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                                                LinearLayout.LayoutParams.MATCH_PARENT)
        blockLayout.layoutParams = lp

        mWinManager = getSystemService(WINDOW_SERVICE) as WindowManager
        mParams = WindowManager.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.OPAQUE
        )
        mParams.gravity = Gravity.LEFT or Gravity.TOP
        mParams.alpha = 0.3F

        userImageDialog = UserImageDialog(this, null) // make empty dialog to prevent crash

        ctrlBtn = binding.fabTestControl
        ctrlBtn.setOnClickListener {
            toggleControl()
        }

        galleryBtn = binding.fabTestGallery
        galleryBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*")
            startActivityResult.launch(intent)
        }


        backCtrlBtn = binding.fabTestBack
        backCtrlBtn.setOnClickListener {
            val mainIntent = Intent(this, MainActivity::class.java).apply {
                Intent.ACTION_VIEW
            }
            startActivity(mainIntent)
            finish()
        }

        tester = NNImageTester(DataShared.testFolder, DataShared.detModelPointer, DataShared.recModelPointer){
            Log.i(TAG, "Test ended")
            isOverLayed.set(false)
            mWinManager.removeView(blockLayout)
            Log.i(TAG, "Removed transparent window overlay")
            if(userImageDialog.isShowing)
                userImageDialog.dismiss()

            val testResultIntent = Intent(this, TestActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("action", "GoForward")
            }
            startActivity(testResultIntent)
        }

        startCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)

        var action = ""
        if (intent != null) {
            action = intent.getStringExtra("action").toString()
        }

        Handler(Looper.getMainLooper()).postDelayed({
            when(action) {
                "GoForward" -> {
                    val infIntent = Intent(this, InferenceActivity::class.java).apply {
                        Intent.ACTION_VIEW
                    }
                    startActivity(infIntent)
                }
            }
            finish()
        }, 2000)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun startCamera() {
        val cameraController = LifecycleCameraController(baseContext)
        previewView = binding.testCameraPreview

        previewView.setOnTouchListener { v: View, e: MotionEvent ->
            if (e.action == MotionEvent.ACTION_UP && isOverLayed.get() == false) {
                isOverLayed.set(true) // blocking duplicated touch event
                previewView.bitmap?.let{ createJpegFile(it) }

                Log.i(TAG, "Test starting")
                thread(start = true) {
                    tester.test()
                }
                mWinManager.addView(blockLayout, mParams)
                Log.i(TAG, "Added transparent window overlay")
            }
            true
        }

        cameraController.bindToLifecycle(this)
        previewView.controller = cameraController
    }

    @SuppressLint("SimpleDateFormat")
    @Throws(IOException::class)
    private fun createJpegFile(bitmap: Bitmap): String {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val dir = File(DataShared.testFolder+"/test")
        var imagePath:String = ""

        dir.mkdir()
        val file = File.createTempFile("${timeStamp}_", ".jpg", dir)
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
    fun copyFile(oneFile: File, destDir: File){
        if (!destDir.exists()) {
            destDir.mkdir()
        }

        val fInputStream = oneFile.inputStream()
        val newFile = File(destDir.toString() + "/" + oneFile.name)
        val out: OutputStream = FileOutputStream(newFile)
        val buffer = ByteArray(1024)
        var read: Int
        while (fInputStream.read(buffer).also { read = it } != -1) {
            out.write(buffer, 0, read)
        }
        fInputStream.close()
        out.close()
    }

    fun getPathFromContentUri(uri: Uri): String{
        var columnIndex = 0
        val pro = arrayOf(MediaStore.Images.Media.DATA)
        val cur = contentResolver.query(uri, pro, null, null, null)
        if(cur!!.moveToFirst())
            columnIndex = cur.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)

        val res = cur.getString(columnIndex)
        cur.close()
        return res
    }

    fun toggleControl(){
        if(!fabControlOpened) {
            ctrlBtn.setImageResource(R.drawable.controller_extract)
            val test_anim = ObjectAnimator.ofFloat(galleryBtn, "translationY", -200f)
            test_anim.duration = 200
            test_anim.start()
            val back_anim = ObjectAnimator.ofFloat(backCtrlBtn, "translationY", -400f)
            back_anim.duration = 200
            back_anim.start()
        } else {
            ctrlBtn.setImageResource(R.drawable.controller_add)
            val test_anim = ObjectAnimator.ofFloat(galleryBtn, "translationY", 0f)
            test_anim.duration = 200
            test_anim.start()
            val back_anim = ObjectAnimator.ofFloat(backCtrlBtn, "translationY", 0f)
            back_anim.duration = 200
            back_anim.start()
        }
        fabControlOpened = !fabControlOpened
    }
}
