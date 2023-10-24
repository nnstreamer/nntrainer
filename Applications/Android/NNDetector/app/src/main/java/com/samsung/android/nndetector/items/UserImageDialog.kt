// SPDX-License-Identifier: Apache-2.0
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
