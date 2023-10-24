// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Point
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import com.samsung.android.nndetector.databinding.ActivitySplashBinding
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

@SuppressLint("CustomSplashScreen")
class SplashActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySplashBinding

    companion object {
        private const val TAG = "SplashActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        globalInitialize()

        Handler(Looper.getMainLooper()).postDelayed({
            val mainIntent = Intent(this, MainActivity::class.java).apply {
                Intent.ACTION_VIEW
            }
            startActivity(mainIntent)
            finish()
        }, 3000)
    }

    fun globalInitialize(){
        DataShared.dispMaxHeight = getDisplayMaxHeight() // get max height of display
        DataShared.dispMaxWidth = getDisplayMaxWidth()

        DataShared.rootFolder = applicationContext.filesDir.path
        DataShared.dataFolder = DataShared.rootFolder + "/Data"

        DataShared.trainFolder = DataShared.dataFolder + "/data/train"
        DataShared.makeFolders(DataShared.trainFolder)
//        copyAssetFolder("data/train", trainFolder)

        DataShared.testFolder = DataShared.dataFolder + "/data/test"
        DataShared.makeFolders(DataShared.testFolder)
//        copyAssetFolder("data/test", testFolder)

        DataShared.recBackbonesFolder = DataShared.dataFolder + "/backbones/recognition"
        DataShared.makeFolders(DataShared.recBackbonesFolder)
        copyAssetFolder("backbones/recognition", DataShared.recBackbonesFolder)

        DataShared.detBackbonesFolder = DataShared.dataFolder + "/backbones/detection"
        DataShared.makeFolders(DataShared.detBackbonesFolder)
        copyAssetFolder("backbones/detection", DataShared.detBackbonesFolder)

        DataShared.detFolder = DataShared.dataFolder + "/Inference/Detection"
        DataShared.makeFolders(DataShared.detFolder + "/Test")

        DataShared.recFolder = DataShared.dataFolder + "/Inference/Recognition"
        DataShared.makeFolders(DataShared.recFolder + "/Test")

        if(File(DataShared.dataFolder + "/data/data.properties").createNewFile()){
            DataShared.storeFakePropeties("ttMap", "{fake=Empty}")
        }
        DataShared.loadMaps() // initialize map from properties
    }

    fun copyAssetFolder(srcName: String, dstName: String): Boolean {
        return try {
            var result = true
            val fileList = assets.list(srcName) ?: return false
            if (fileList.size == 0) {
                result = copyAssetFile(srcName, dstName)
            } else {
                val file = File(dstName)
                result = file.mkdirs()
                for (filename in fileList) {
                    result = result and copyAssetFolder(
                        srcName + File.separator + filename,
                        dstName + File.separator + filename
                    )
                }
            }
            result
        } catch (e: IOException) {
            e.printStackTrace()
            false
        }
    }

    fun copyAssetFile(srcName: String?, dstName: String?): Boolean {
        return try {
            val `in` = assets.open(srcName!!)
            val outFile = dstName?.let { File(it) }
            val out: OutputStream = FileOutputStream(outFile)
            val buffer = ByteArray(1024)
            var read: Int
            while (`in`.read(buffer).also { read = it } != -1) {
                out.write(buffer, 0, read)
            }
            `in`.close()
            out.close()
            true
        } catch (e: IOException) {
            e.printStackTrace()
            false
        }
    }

    private fun getDisplayMaxHeight(): Int{
        val p = Point()
        val q = Point()
        display?.getCurrentSizeRange(p, q)
        return q.x
    }

    private fun getDisplayMaxWidth(): Int{
        val p = Point()
        val q = Point()
        display?.getCurrentSizeRange(p, q)
        return p.x
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}
