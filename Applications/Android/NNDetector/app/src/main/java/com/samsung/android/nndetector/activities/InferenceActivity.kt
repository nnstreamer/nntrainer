// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.samsung.android.nndetector.databinding.ActivityInferenceBinding
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.properties.Delegates

class InferenceActivity: AppCompatActivity() {
    private lateinit var binding: ActivityInferenceBinding
    private lateinit var infStartBtn: FloatingActionButton
    private lateinit var infStopBtn: FloatingActionButton
    private lateinit var backBtn: FloatingActionButton
    private lateinit var forwardBtn: FloatingActionButton
    private lateinit var previewView: PreviewView
    private lateinit var cameraController: LifecycleCameraController

    private lateinit var isAnalyzerEnded: AtomicBoolean
    private var notFoundCnt by Delegates.notNull<Int>()

    private lateinit var inferenceResultDialog: InferenceResultDialog

    companion object {
        private const val TAG = "InferenceActivity"
    }

    init{
        isAnalyzerEnded = AtomicBoolean(false)
        notFoundCnt = 0
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityInferenceBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.hide()

        infStartBtn = binding.fabInferenceStart
        infStartBtn.setOnClickListener {
            isAnalyzerEnded.set(false)
            infStopBtn.isEnabled = true
            infStartBtn.isEnabled = false
            forwardBtn.isEnabled = false
            backBtn.isEnabled = false
            cameraController.setImageAnalysisAnalyzer(
                ContextCompat.getMainExecutor(this),
                NNImageAnalyzer(DataShared.detModelPointer, DataShared.recModelPointer, DataShared.detFolder,
                    DataShared.recFolder, DataShared.dispMaxHeight) { results ->
                    if(isAnalyzerEnded.get())
                        return@NNImageAnalyzer

                    if(results.size == 0)
                        return@NNImageAnalyzer

                    var itemDrawables:Array<ItemDrawable> = arrayOf()
                    var personObjectFound = false
                    for(result in results) {
                        if(!personObjectFound && DataShared.personalObjectList.indexOf(result.label) != -1){
                            personObjectFound = true
                            notFoundCnt = 0
                        }

                        val itemDrawable = result.let { ItemDrawable(result.rect, result.label,
                            result.score, Color.CYAN) }
                        itemDrawables += itemDrawable
                    }

                    if(personObjectFound == false){
                        notFoundCnt++
                    }
                    if(notFoundCnt > 10){
                        notFoundCnt = 0

                        val stopIntent = Intent(this, InferenceActivity::class.java).apply {
                            Intent.ACTION_VIEW
                            putExtra("inferenceCommand", "inferenceStop")
                        }
                        startActivity(stopIntent)
                    }

                    previewView.overlay.clear()
                    for (itemDrawable in itemDrawables) {
                        previewView.overlay.add(itemDrawable)
                    }
                }
            )
        }

        infStopBtn = binding.fabInferenceStop
        infStopBtn.isEnabled = false
        infStopBtn.setOnClickListener {
            isAnalyzerEnded.set(true)
            cameraController.clearImageAnalysisAnalyzer()
            previewView.overlay.clear()

            infStopBtn.isEnabled = false
            infStartBtn.isEnabled = true
            forwardBtn.isEnabled = true
            backBtn.isEnabled = true
        }

        backBtn = binding.fabGoBack
        backBtn.setOnClickListener {
            val testIntent = Intent(this, TestActivity::class.java).apply {
                Intent.ACTION_VIEW
            }
            startActivity(testIntent)
            finish()
        }

        forwardBtn = binding.fabGoForward
        forwardBtn.setOnClickListener {
            val mainIntent = Intent(this, MainActivity::class.java).apply {
                Intent.ACTION_VIEW
            }
            startActivity(mainIntent)
            finish()
        }

        DataShared.updatePersonalObjectList()

        startCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)

        if (intent != null) {
            if(intent.getStringExtra("inferenceCommand") == "inferenceStop") {
                infStopBtn.performClick()
                forwardBtn.isEnabled = true
                backBtn.isEnabled = true
                inferenceResultDialog = InferenceResultDialog(this)
                inferenceResultDialog.show()
            }
        }
    }

    private fun startCamera() {
        cameraController = LifecycleCameraController(baseContext)
        previewView = binding.infCameraPreview

        cameraController.bindToLifecycle(this)
        previewView.controller = cameraController
    }
}
