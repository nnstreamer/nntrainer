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

#include "resnet_jni.h"
#include "resnet.h"
#include <android/bitmap.h>

int cur_epoch = 0;
float val_accu = 0.0;

JNIEXPORT jlong JNICALL
Java_com_applications_resnetjni_MainActivity_createModel(JNIEnv *env,
                                                         jobject j_obj,
                                                         jstring input_shape,
                                                         jint unit) {
  const char *in_shape = env->GetStringUTFChars(input_shape, 0);
  std::string in(in_shape);
  ml::train::Model *model_ = createResnet18(in, unit);
  return reinterpret_cast<jlong>(model_);
}

JNIEXPORT jint JNICALL
Java_com_applications_resnetjni_MainActivity_train_1resnet(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong model_pointer) {
  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }

  ml::train::Model *model_ =
    reinterpret_cast<ml::train::Model *>(model_pointer);

  jint status = init(argc, argv, model_);

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return status;
}

JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_testResnet(JNIEnv *env,
                                                        jobject j_obj,
                                                        jobjectArray args,
                                                        jlong model_pointer) {
  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }

  ml::train::Model *model_ =
    reinterpret_cast<ml::train::Model *>(model_pointer);

  std::string result = testModel(argc, argv, model_);
  jstring ret = (env)->NewStringUTF(result.c_str());

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return ret;
}

JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_inferResnet(JNIEnv *env,
                                                         jobject j_obj,
                                                         jobjectArray args,
                                                         jobject bmp,
                                                         jlong model_pointer) {
  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }

  ml::train::Model *model_ =
    reinterpret_cast<ml::train::Model *>(model_pointer);

  int rcc;
  jboolean rc = JNI_FALSE;

  AndroidBitmapInfo info;

  uint8_t *pBmp = NULL;

  try {
    rcc = AndroidBitmap_getInfo(env, bmp, &info);
    if (rcc != ANDROID_BITMAP_RESULT_SUCCESS) {
      throw "get Bitmap Info failure";
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
      throw "only ARGB888 format support";
    }

    rcc = AndroidBitmap_lockPixels(env, bmp, (void **)&pBmp);

    if (rcc != ANDROID_BITMAP_RESULT_SUCCESS) {
      throw "lockPixels failure";
    }

    for (int y = 0; y < info.height; y++) {
      uint8_t *px = pBmp + y * info.stride;
      for (int x = 0; x < info.width; x++) {

        px[0] = uint8_t(((float)x / info.width) * 255.0f);  // R
        px[1] = uint8_t(((float)y / info.height) * 255.0f); // G
        px[2] = 0x00;                                       // B
        px[3] = 0xff;                                       // A

        px += 4;
      }
    }

    rc = JNI_TRUE;
  } catch (const char *e) {
  }

  if (pBmp) {
    AndroidBitmap_unlockPixels(env, bmp);
  }

  std::string result = inferModel(argc, argv, pBmp, model_);
  jstring ret = (env)->NewStringUTF(result.c_str());

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return ret;
}

JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_getTrainingStatus(
  JNIEnv *env, jobject j_obj, jlong model_pointer, jint cur_iter,
  jint batch_size) {
  int status = 0;

  ml::train::Model *model_ =
    reinterpret_cast<ml::train::Model *>(model_pointer);
  ml::train::RunStats train_stat = model_->getTrainingStats();
  ml::train::RunStats valid_stat = model_->getValidStats();

  jstring ret = (env)->NewStringUTF("-");
  if (cur_iter != train_stat.num_iterations && train_stat.num_iterations != 0) {
    std::string s =
      "#" + std::to_string(train_stat.epoch_idx) + "/" +
      std::to_string(train_stat.max_epoch) +
      displayProgress(train_stat.num_iterations,
                      train_stat.loss / train_stat.num_iterations, batch_size) +
      "\n";

    ret = (env)->NewStringUTF(s.c_str());
  }

  cur_epoch = train_stat.epoch_idx;

  if (val_accu != valid_stat.accuracy && valid_stat.accuracy != 0.0) {
    std::string s = "#" + std::to_string(train_stat.epoch_idx) + "/" +
                    std::to_string(train_stat.max_epoch) +
                    " [ Accuracy : " + std::to_string(valid_stat.accuracy) +
                    "% - Validation Loss : " + std::to_string(valid_stat.loss) +
                    " ]\n";
    val_accu = valid_stat.accuracy;

    ret = (env)->NewStringUTF(s.c_str());
  }

  return ret;
}

JNIEXPORT jint JNICALL
Java_com_applications_resnetjni_MainActivity_getCurrentEpoch(
  JNIEnv *env, jobject j_obj, jlong model_pointer) {
  int status = 0;

  ml::train::Model *model_ =
    reinterpret_cast<ml::train::Model *>(model_pointer);
  ml::train::RunStats train_stat = model_->getTrainingStats();

  jint epoch_idx = train_stat.epoch_idx;

  return epoch_idx;
}

JNIEXPORT jint JNICALL Java_com_applications_resnetjni_MainActivity_requestStop(
  JNIEnv *env, jobject j_obj) {
  setStop();
  return 0;
}

JNIEXPORT jstring JNICALL
Java_com_applications_resnetjni_MainActivity_getTestingResult(JNIEnv *env,
                                                              jobject j_obj) {

  jstring ret = (env)->NewStringUTF(getTestingStatus().c_str());

  return ret;
}

JNIEXPORT jboolean JNICALL
Java_com_applications_resnetjni_MainActivity_modelDestroyed(JNIEnv *env,
                                                            jobject j_obj) {

  return modelDestroyed();
}
