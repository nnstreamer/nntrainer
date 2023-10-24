// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.util.Properties
import kotlin.properties.Delegates

class DataShared {
    companion object{
        private const val TAG = "DataShared"

        val detInputImgDim = 640
        val recInputImgDim = 224
        val numClass = 10
        val knnNormVariant = "UN"
        const val detAnchorNum = 8400
        const val detOutDim = 14

        lateinit var rootFolder: String
        lateinit var dataFolder: String
        lateinit var trainFolder: String
        lateinit var testFolder: String

        lateinit var detFolder: String
        lateinit var recFolder: String
        lateinit var detBackbonesFolder: String
        lateinit var recBackbonesFolder: String
        var detModelPointer by Delegates.notNull<Long>()
        var recModelPointer by Delegates.notNull<Long>()
        var dispMaxHeight by Delegates.notNull<Int>()
        var dispMaxWidth by Delegates.notNull<Int>()

        val yoloLabels = arrayOf("pen", "mug", "bottle", "book", "glasses", "watch", "mouse", "keyboard", "fruit", "snack")
//        var labelMap: Map<String, String> = mapOf()
        var ttMap: Map<String, String> = mapOf()
        var isTrained = false

        var personalObjectList: MutableList<String> = mutableListOf()

        fun loadMaps(){
            Log.i(TAG, "reload Maps")
            val prop = Properties()
            prop.load(File(rootFolder + "/Data/data/data.properties").inputStream())

            val strTTMap:String = prop.getProperty("ttMap")
                        .substringAfter("{")
                        .substringBeforeLast("}")
            ttMap = strTTMap.split(",").associate {
                val labels = it.split("=")
                labels[0].trim() to labels[1]
            }
            ttMap -= "fake"
            Log.i(TAG, ttMap.toString())

//            val strLblMap:String = prop.getProperty("labelMap")
//                        .substringAfter("{")
//                        .substringBeforeLast("}")
//            labelMap = strLblMap.split(",").associate {
//                val labels = it.split("=")
//                labels[0].trim() to labels[1]
//            }
//            labelMap -= "fake"
//            Log.i(TAG, labelMap.toString())
        }

        fun storeFakePropeties(key: String, value: String){
            val prop = Properties()
            prop.load(File(rootFolder + "/Data/data/data.properties").inputStream())
            prop.setProperty(key, value)

            val out = FileOutputStream(File(rootFolder + "/Data/data/data.properties"))
            prop.store(out, "Initial Fake Data")
        }

        fun storeTTMap(){
            Log.i(TAG, "store TTMap: ${ttMap}")
            val prop = Properties()
            prop.load(File(rootFolder + "/Data/data/data.properties").inputStream())
            prop.setProperty("ttMap", ttMap.toString())

            val out = FileOutputStream(File(rootFolder + "/Data/data/data.properties"))
            prop.store(out, "Maps")
        }

//        fun storeLabelMap(context: Context){
//            Log.i(TAG, "store LabelMap: ${labelMap}")
//            val prop = Properties()
//            prop.load(File(context.filesDir.path + "/data.properties").inputStream())
//            prop.setProperty("labelMap", labelMap.toString())
//
//            val out = FileOutputStream(File(context.filesDir.path + "/data.properties"))
//            prop.store(out, "Maps")
//        }

        fun makeFolders(dirPath: String){
            val needToMakeFolder = dirPath.substringAfter(rootFolder+"/")
            val folders: List<String> = needToMakeFolder.split("/")

            var makeDirPath = rootFolder
            for(folder in folders){
                makeDirPath += "/" + folder
                val nFolder: File = File(makeDirPath)
                val result = nFolder.mkdir()
                if (result) {
                    Log.d(TAG, "$nFolder was created")
                } else {
                    Log.d(TAG, "Failed to create $nFolder folder")
                }
            }
        }

        fun updatePersonalObjectList() {
            val trainDir = File(trainFolder)
            for(dir in trainDir.listFiles()!!)
                personalObjectList += dir.name
        }

        fun deleteFolder(dirPath: String){
            File(dirPath).deleteRecursively()
        }
    }
}
