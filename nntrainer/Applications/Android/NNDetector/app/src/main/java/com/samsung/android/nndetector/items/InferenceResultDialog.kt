// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. HS.Kim <hs0207.kim@samsung.com>
 * 
 * @file   InferenceResultDialog.kt
 * @date   24 Oct 2023
 * @brief  Showing the final conclusion of inference
 * @author HS.Kim (hs0207.kim@samsung.com)
 * @bug    No known bugs
 */
package com.samsung.android.nndetector

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import com.samsung.android.nndetector.databinding.InferenceResultDialogBinding

class InferenceResultDialog(context: Context): Dialog(context)  {
    private lateinit var binding: InferenceResultDialogBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = InferenceResultDialogBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.inferenceResult.text = "Sorry, I couldn't detect any object here. You can choose 2 options.\n\n Option1: acquire more training, choose '>' button\n\n Option2: try again, choose '<' button"
    }
}
