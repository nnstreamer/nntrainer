// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.samsung.android.nndetector.databinding.ActivityTrainBinding
import java.io.File
import java.util.Timer
import kotlin.concurrent.thread
import kotlin.concurrent.timer

class TrainActivity: ComponentActivity(){
    private lateinit var binding: ActivityTrainBinding
    private lateinit var needToTrain: String
    private lateinit var trainer: NNImageTrainer
//    private lateinit var trainTimer: Timer

    companion object {
        private const val TAG = "TrainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTrainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        needToTrain = intent.getStringExtra("needToTrain").toString()

        if(!existSubDirAndFiles(File(DataShared.trainFolder))) {
            Log.i(TAG, "Train dataset is empty")
            val trainResultIntent = Intent(this, TrainActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("action", "GoBack")
                putExtra("trainResult", "DataSet for Train is empty. Go Back!!")
            }
            startActivity(trainResultIntent)
            return
        }

        if(needToTrain == "No"){
            val trainResultIntent = Intent(this, TrainActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("action", "GoBack")
                putExtra("trainResult", "There is no new Dataset. Please, Add new Dataset.")
            }
            startActivity(trainResultIntent)
            return
        }

        trainer = NNImageTrainer(DataShared.trainFolder, DataShared.detModelPointer, DataShared.recModelPointer){state:NNImageTrainer.TrainerState ->
            Log.i(TAG, "Train ended")
//            trainTimer.cancel()
            val trainResultIntent = Intent(this, TrainActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("action", "GoForward")
                putExtra("trainResult", "I've learned your personal objects. Please, test me acquiring some pictures of your new objects.")
            }
            startActivity(trainResultIntent)
        }

        Log.i(TAG, "Train starting")
//        trainTimer = timer(period = 7000){
//            runOnUiThread {
//                Toast.makeText(this@TrainActivity, "Train in progress...", Toast.LENGTH_SHORT).show()
//            }
//        }

        binding.ttState.text =  "Please allow a few moments while I learn to recognize your personal objects."
        thread(start = true) {
            trainer.train()
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)

        var action = ""
        if (intent != null) {
            binding.ttState.text = intent.getStringExtra("trainResult")
            action = intent.getStringExtra("action").toString()
        }

        Handler(Looper.getMainLooper()).postDelayed({
            when(action) {
                "GoBack" -> {
                    val mainIntent = Intent(this, MainActivity::class.java).apply {
                        Intent.ACTION_VIEW
                    }
                    startActivity(mainIntent)
                }

                "GoForward" -> {
                    val testIntent = Intent(this, TestActivity::class.java).apply {
                        Intent.ACTION_VIEW
                    }
                    startActivity(testIntent)
                }
            }
            finish()
        }, 3000)
    }

    private fun existSubDirAndFiles(trainDir: File): Boolean {
        val subDirs = trainDir.listFiles()
        if(subDirs!!.isNotEmpty()) {
            for (dir in subDirs) {
                val files = dir.listFiles()
                if (files!!.isNotEmpty()) {
                    return true
                }
            }
        }

        return false
    }
}
