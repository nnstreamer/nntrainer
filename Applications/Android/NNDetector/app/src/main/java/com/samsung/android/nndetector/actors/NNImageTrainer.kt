// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.util.Log
import kotlin.properties.Delegates

class NNImageTrainer(
    trainFolderPath: String, detModePointer: Long, recModelPointer: Long,
    private val listener: ((state: TrainerState) -> Unit)?) {
    private lateinit var mTrainFolderPath: String
    private var mDetModelPointer by Delegates.notNull<Long>()
    private var mRecModelPointer by Delegates.notNull<Long>()

    companion object{
        private const val TAG:String = "NNImageTrainer"
    }

    init {
        mTrainFolderPath = trainFolderPath
        mDetModelPointer = detModePointer
        mRecModelPointer = recModelPointer
    }

    external fun trainPrototypes(modelPath: Array<String>, detModelPointer: Long, recModelPointer: Long) : String

    fun train(){
        val program_name = "NNDetector"

        Log.i(TAG, "Start training prototypes ")
        val trainArgs =
            arrayOf<String>(program_name, mTrainFolderPath, DataShared.detInputImgDim.toString(),
                DataShared.detOutDim.toString(), DataShared.detAnchorNum.toString(),
                DataShared.recInputImgDim.toString(), DataShared.numClass.toString())
        trainPrototypes(trainArgs, mDetModelPointer, mRecModelPointer)
        Log.i(TAG, "Finished training prototypes ")

        val state = TrainerState()
        state.setData("Train done")

        DataShared.isTrained = true

        listener?.let { it(state) }
    }

    fun initialTrain(){
        val program_name = "NNDetector"

        Log.i(TAG, "Start initial training prototypes ")
        val trainArgs =
            arrayOf<String>(program_name, mTrainFolderPath, DataShared.detInputImgDim.toString(),
                DataShared.detOutDim.toString(), DataShared.detAnchorNum.toString(),
                DataShared.recInputImgDim.toString(), DataShared.numClass.toString())
        trainPrototypes(trainArgs, mDetModelPointer, mRecModelPointer)
        Log.i(TAG, "Finished initial training prototypes ")

        DataShared.isTrained = true
    }

    fun done(){
    }

    class TrainerState{
        var trainInfo: String
        init{
            trainInfo = "initial empty value"
        }
        fun setData(state: String){
            trainInfo = ""
            trainInfo = state
        }
    }
}
