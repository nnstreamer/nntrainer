// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <jni.h>

#ifndef _Included_com_applications_resnetjni_MainActivity
#define _Included_com_applications_resnetjni_MainActivity
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
Java_com_applications_resnetjni_MainActivity_createModel(JNIEnv *, jobject,
                                                         jstring, jint);

/**
 * @brief Train Model
 * @param jobjectArray Model argument list
 * @param jlong Model pointer
 * @return status
 */
JNIEXPORT jint JNICALL
Java_com_applications_resnetjni_MainActivity_train_1resnet(JNIEnv *, jobject,
                                                           jobjectArray, jlong);

/**
 * @brief Test Model
 * @param jobjectArray Model argument list
 * @param jlong Model pointer
 * @return string test results as a string
 */
JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_testResnet(JNIEnv *, jobject,
                                                        jobjectArray, jlong);

/**
 * @brief Inference Model
 * @param jlong Model pointer
 * @param jobeject bmp from android java
 * @return string inference result
 */
JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_inferResnet(JNIEnv *, jobject,
                                                         jobjectArray,
                                                         jobject bmp, jlong);

/**
 * @brief Training Status Getter
 * @param jlong Model pointer
 * @param jint current iteration
 * @param jint batch size
 * @return string training status
 */
JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_getTrainingStatus(JNIEnv *,
                                                               jobject, jlong,
                                                               jint, jint);

/**
 * @brief Current Epoch Getter
 * @param jlong Model pointer
 * @return int Current Epoch
 */
JNIEXPORT jint JNICALL
Java_com_applications_resnetjni_MainActivity_getCurrentEpoch(JNIEnv *, jobject,
                                                             jlong);

/**
 * @brief Stop
 * @return jint
 */
JNIEXPORT jint JNICALL
Java_com_applications_resnetjni_MainActivity_requestStop(JNIEnv *, jobject);

/**
 * @brief Test Result Getter
 * @return int Current Epoch
 */
JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_getTestingResult(JNIEnv *,
                                                              jobject);

/**
 * @brief check model destoryed
 * @return bool true if model is destoryed successfully
 */
JNIEXPORT jboolean JNICALL
Java_com_applications_resnetjni_MainActivity_modelDestroyed(JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
