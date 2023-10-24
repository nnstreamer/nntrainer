// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. Umberto Michieli <u.michieli@samsung.com>.
 * Copyright 2023. Kirill Paramonov <k.paramonov@samsung.com>.
 * Copyright 2023. Mete Ozay <m.ozay@samsung.com>.
 * Copyright 2023. JIJOONG MOON <jijoong.moon@samsung.com>.
 * Copyright 2023. HS.Kim <hs0207.kim@samsung.com>
 *
 * @file   simpleshot_jni.h
 * @date   24 Oct 2023
 * @brief  JNI apis for image recognition/detection
 * @author Umberto Michieli(u.michieli@samsung.com)
 * @author Kirill Paramonov(k.paramonov@samsung.com)
 * @author Mete Ozay(m.ozay@samsung.com)
 * @author JIJOONG MOON(jijoong.moon@samsung.com)
 * @author HS.Kim(hs0207.kim@samsung.com)
 * @bug    No known bugs
 */
#include <jni.h>

#ifndef _Included_com_applications_simpleshotjni_MainActivity
#define _Included_com_applications_simpleshotjni_MainActivity
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create Recognition Model
 *
 * @param jobjectArray Model argument list
 * @return model pointer
 */
JNIEXPORT jlong JNICALL
Java_com_samsung_android_nndetector_MainActivity_createRecognitionModel(
  JNIEnv *env, jobject j_obj, jobjectArray args);

/**
 * @brief Create Detection Model
 *
 * @param jobjectArray Model argument list
 * @return model pointer
 */
JNIEXPORT jlong JNICALL
Java_com_samsung_android_nndetector_MainActivity_createDetectionModel(
  JNIEnv *env, jobject j_obj, jobjectArray args);

/**
 * @brief Train Model
 * @param jobjectArray Model argument list
 * @param jlong Model pointer
 * @return status
 */
JNIEXPORT jint JNICALL
Java_com_samsung_android_nndetector_NNImageTrainer_trainPrototypes(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong det_model_pointer,
  jlong rec_model_pointer);

/**
 * @brief Test Model
 * @param jobjectArray Model argument list
 * @param jlong Model pointer
 * @return string test results as a string
 */
JNIEXPORT jstring JNICALL
Java_com_samsung_android_nndetector_NNImageTester_testPrototypes(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong det_model_pointer,
  jlong rec_model_pointer);

/**
 * @brief Run Object Detection
 * @param jobjectArray Model argument list
 * @param jlong Model pointer
 * @return string detection results as a string
 */
JNIEXPORT jstring JNICALL
Java_com_samsung_android_nndetector_NNImageAnalyzer_runDetector(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong det_model_pointer);

#ifdef __cplusplus
}
#endif
#endif
