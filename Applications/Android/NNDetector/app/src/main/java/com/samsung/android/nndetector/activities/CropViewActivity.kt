// SPDX-License-Identifier: Apache-2.0
package com.samsung.android.nndetector

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.samsung.android.nndetector.databinding.CropViewBinding
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

class CropViewActivity: ComponentActivity() {
    private lateinit var binding: CropViewBinding
    private lateinit var label: String
    private lateinit var cropImagePath: String

    companion object {
        private const val TAG = "CropViewActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = CropViewBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.croppedImage.setImageURI(Uri.parse(intent.getStringExtra("cropImagePath")))
        label = intent.getStringExtra("label").toString()
        binding.labelValue.text = label
        binding.score.text = intent.getStringExtra("score")
        binding.textUserAction.append(label)

        cropImagePath = intent.getStringExtra("cropImagePath").toString()
        val tempFolderPath = cropImagePath.substringBeforeLast("/")

        val saveBtn = binding.save
        saveBtn.setOnClickListener{
            if(!binding.inputEdit.text.isEmpty() and
                !DataShared.yoloLabels.contains(binding.inputEdit.text.toString())) {
                Log.i(TAG, "ttMap[" + binding.inputEdit.text.toString() + "] to ${label}")
                DataShared.ttMap += binding.inputEdit.text.toString() to label
                label = binding.inputEdit.text.toString()

                DataShared.storeTTMap()
            }

            Toast.makeText(
                this@CropViewActivity,
                "saved as \"$label\"",
                Toast.LENGTH_SHORT
            ).show()

            val mainIntent = Intent(this, MainActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("newDataset", "Yes")
            }
            startActivity(mainIntent)

            val mTrainFolderPath = tempFolderPath.substringBeforeLast("/")+"/train"
            DataShared.makeFolders(mTrainFolderPath+ "/"+label)
            copyFile(File(cropImagePath), File(mTrainFolderPath+ "/"+label))

            DataShared.deleteFolder(tempFolderPath)
            finish()
        }

        val cancelBtn = binding.cancel
        cancelBtn.setOnClickListener{
            if(!binding.inputEdit.text.isEmpty())
                label = binding.inputEdit.text.toString()

            Toast.makeText(
                this@CropViewActivity,
                "canceled to saving \"$label\"",
                Toast.LENGTH_SHORT
            ).show()

            val mainIntent = Intent(this, MainActivity::class.java).apply {
                Intent.ACTION_VIEW
                putExtra("newDataset", "No")
            }
            startActivity(mainIntent)

            DataShared.deleteFolder(tempFolderPath)
            finish()
        }
    }

    private fun copyFile(oneFile: File, destDir: File){
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

    override fun onDestroy() {
        super.onDestroy()
    }

    private fun hideKeyboard(){
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(binding.inputEdit.windowToken, 0)
    }
}
