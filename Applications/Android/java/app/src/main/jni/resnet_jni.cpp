//
// Created by hs89lee on 23. 2. 13..
//

#include "resnet.h"
#include "resnet_jni.h"

JNIEXPORT jint JNICALL
Java_com_applications_resnetjni_MainActivity_train_1resnet(JNIEnv *env,
                                                           jobject j_obj,
                                                           jobjectArray args) {
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

  int status = init(argc, argv);
  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return status;
}
