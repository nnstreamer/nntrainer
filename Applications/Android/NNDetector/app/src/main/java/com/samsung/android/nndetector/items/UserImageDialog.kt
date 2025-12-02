// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. HS.Kim <hs0207.kim@samsung.com>
 * 
 * @file   UserImageDialog.kt
 * @date   24 Oct 2023
 * @brief  Showing image of gallery
 * @author HS.Kim (hs0207.kim@samsung.com)
 * @bug    No known bugs
 */
package com.samsung.android.nndetector

import android.app.Dialog
import android.content.Context
import android.net.Uri
import android.os.Bundle
import com.samsung.android.nndetector.databinding.UserDialogBinding

class UserImageDialog(context: Context, imageUri: Uri?): Dialog(context) {
    private lateinit var mImageUri:Uri
    private lateinit var binding:UserDialogBinding

    init{
        if (imageUri != null) {
            mImageUri = imageUri
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = UserDialogBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.userImage.setImageURI(mImageUri)
    }
}
