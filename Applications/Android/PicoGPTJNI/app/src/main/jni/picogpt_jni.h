// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move picogpt model creating to separate sourcefile
 * @brief  task runner for the picogpt
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <jni.h>

#ifndef _Included_com_applications_picogptjni_MainActivity
#define _Included_com_applications_picogptjni_MainActivity
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create Model
 *
 * @param jint number of output ( number of classes )
 * @return model pointer
 */
JNIEXPORT jlong JNICALL
Java_com_applications_picogptjni_MainActivity_createModel(JNIEnv *, jobject);

/**
 * @brief Inference Model
 * @param jlong Model pointer
 * @param jobeject bmp from android java
 * @return string inference result
 */
JNIEXPORT jstring JNICALL
Java_com_applications_picogptjni_MainActivity_inferPicoGPT(JNIEnv *, jobject,
                                                           jstring, jstring,
                                                           jlong);
/**
 * @brief check model destoryed
 * @return bool true if model is destoryed successfully
 */
JNIEXPORT jboolean JNICALL
Java_com_applications_picogptjni_MainActivity_modelDestroyed(JNIEnv *, jobject);

JNIEXPORT jstring JNICALL
Java_com_applications_picogptjni_MainActivity_getInferResult(JNIEnv *,
							     jobject);

#ifdef __cplusplus
}
#endif
#endif
