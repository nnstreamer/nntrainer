// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.util.Log
import kotlin.properties.Delegates

class NNImageTester(
    testFolderPath: String, detModePointer: Long, recModelPointer: Long,
    private val listener: ((state: TesterState) -> Unit)? ) {
    private lateinit var mTestFolderPath: String
    private var mDetModelPointer by Delegates.notNull<Long>()
    private var mRecModelPointer by Delegates.notNull<Long>()

    companion object{
        private const val TAG:String = "NNImageTester"
    }

    init {
        mTestFolderPath = testFolderPath
        mDetModelPointer = detModePointer
        mRecModelPointer = recModelPointer
    }

    external fun testPrototypes(modelPath: Array<String>, detModelPointer: Long, recModelPointer: Long) : String

    fun test(){
        val program_name = "NNDetector"

        Log.d(TAG, "Start testing prototypes ")
        val testArgs =
            arrayOf<String>(program_name, mTestFolderPath, DataShared.detInputImgDim.toString(),
                DataShared.detOutDim.toString(), DataShared.detAnchorNum.toString(),
                DataShared.recInputImgDim.toString(), DataShared.numClass.toString())
        val ret = testPrototypes(testArgs, mDetModelPointer, mRecModelPointer)
        Log.i(TAG, "Finished testing prototypes ")

        val strList: List<String> = ret.split("\n\n")
        for (i in 0 until strList.size) {
            if(strList[i].isEmpty())
                continue

            val personalLbl = strList[i].substringAfter("label ")
            Log.i(TAG, "${i}'s tested label: ${personalLbl}")
        }

        val state = TesterState()
        state.setData("Test done")

        listener?.let{it(state)}
    }

    fun instantTest(): String{
        val program_name = "NNDetector"

        Log.d(TAG, "Start testing prototypes ")
        val testArgs =
            arrayOf<String>(program_name, mTestFolderPath, DataShared.detInputImgDim.toString(),
                DataShared.detOutDim.toString(), DataShared.detAnchorNum.toString(),
                DataShared.recInputImgDim.toString(), DataShared.numClass.toString())
        val ret = testPrototypes(testArgs, mDetModelPointer, mRecModelPointer)
        Log.i(TAG, "Finished testing prototypes ")

        val strList: List<String> = ret.split("\n\n")
        var personalLbl: String = ""
        for (i in 0 until strList.size) {
            if(strList[i].isEmpty())
                continue

            personalLbl = strList[i].substringAfter("label ")
            Log.i(TAG, "${i}'s tested label: ${personalLbl}")
            break
        }
        return personalLbl
     }

    class TesterState{
        var testInfo: String
        init{
            testInfo = "initial empty value"
        }
        fun setData(state: String){
            testInfo = ""
            testInfo = state
        }
    }
}
