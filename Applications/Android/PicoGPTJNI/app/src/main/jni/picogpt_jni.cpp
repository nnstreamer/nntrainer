// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
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

#include "picogpt_jni.h"
#include "picogpt.h"
#include <android/bitmap.h>

int cur_epoch = 0;
float val_accu = 0.0;

JNIEXPORT jlong JNICALL
Java_com_applications_picogptjni_MainActivity_createModel(JNIEnv *env,
                                                         jobject j_obj) {
  ml::train::Model *model_ = createPicogpt();
  return reinterpret_cast<jlong>(model_);
}

JNIEXPORT jstring JNICALL
Java_com_applications_picogptjni_MainActivity_inferPicoGPT(
  JNIEnv *env, jobject j_obj, jstring path_, jstring text_,
  jlong model_pointer) {
  // const int argc = env->GetArrayLength(args);
  // char **argv = new char *[argc];
  // for (unsigned int i = 0; i < argc; ++i) {
  //   jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
  //   const char *str = env->GetStringUTFChars(j_str, 0);
  //   size_t str_len = strlen(str);
  //   argv[i] = new char[str_len + 1];
  //   strcpy(argv[i], str);
  //   env->ReleaseStringUTFChars(j_str, str);
  // }

  
  const char *str = env->GetStringUTFChars(path_, 0);
  size_t str_len = strlen(str);

  std::string path = std::string(str, str_len);

  str = env->GetStringUTFChars(text_, 0);
  str_len = strlen(str);

  std::string text = std::string(str, str_len);

  ml::train::Model *model_ =
    reinterpret_cast<ml::train::Model *>(model_pointer);

  int rcc;
  jboolean rc = JNI_FALSE;

  AndroidBitmapInfo info;

  std::string result = inferModel(path, text, model_);
  jstring ret = (env)->NewStringUTF(result.c_str());

  // for (unsigned int i = 0; i < argc; ++i) {
  //   delete[] argv[i];
  // }
  // delete[] argv;
  
  return ret;
}

JNIEXPORT jboolean JNICALL
Java_com_applications_picogptjni_MainActivity_modelDestroyed(JNIEnv *env,
                                                            jobject j_obj) {

  return modelDestroyed();
}


JNIEXPORT jstring JNICALL
Java_com_applications_picogptjni_MainActivity_getInferResult(JNIEnv *env,
                                                              jobject j_obj) {

  jstring ret = (env)->NewStringUTF(getInferResult().c_str());

  return ret;
}
