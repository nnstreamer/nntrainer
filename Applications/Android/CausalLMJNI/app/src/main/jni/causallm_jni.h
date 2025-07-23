#ifndef __CAUSALLM_JNI_H__
#define __CAUSALLM_JNI_H__

#include <jni.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create CausalLM model instance
 * @param env JNI environment
 * @param thiz Java object
 * @param config_path Path to model configuration directory
 * @return Model pointer as jlong
 */
JNIEXPORT jlong JNICALL
Java_com_applications_causallmjni_MainActivity_createCausalLMModel(
    JNIEnv *env, jobject thiz, jstring config_path);

/**
 * @brief Initialize the CausalLM model
 * @param env JNI environment
 * @param thiz Java object
 * @param model_pointer Model pointer
 * @return 0 on success, -1 on failure
 */
JNIEXPORT jint JNICALL
Java_com_applications_causallmjni_MainActivity_initializeModel(
    JNIEnv *env, jobject thiz, jlong model_pointer);

/**
 * @brief Load model weights
 * @param env JNI environment
 * @param thiz Java object
 * @param model_pointer Model pointer
 * @param weight_path Path to weight file
 * @return 0 on success, -1 on failure
 */
JNIEXPORT jint JNICALL
Java_com_applications_causallmjni_MainActivity_loadWeights(
    JNIEnv *env, jobject thiz, jlong model_pointer, jstring weight_path);

/**
 * @brief Run inference with CausalLM model
 * @param env JNI environment
 * @param thiz Java object
 * @param model_pointer Model pointer
 * @param input_text Input text prompt
 * @param do_sample Whether to use sampling
 * @return Generated text as jstring
 */
JNIEXPORT jstring JNICALL
Java_com_applications_causallmjni_MainActivity_runInference(
    JNIEnv *env, jobject thiz, jlong model_pointer, jstring input_text, jboolean do_sample);

/**
 * @brief Destroy CausalLM model instance
 * @param env JNI environment
 * @param thiz Java object
 * @param model_pointer Model pointer
 */
JNIEXPORT void JNICALL
Java_com_applications_causallmjni_MainActivity_destroyModel(
    JNIEnv *env, jobject thiz, jlong model_pointer);

#ifdef __cplusplus
}
#endif

#endif /* __CAUSALLM_JNI_H__ */