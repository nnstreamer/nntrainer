// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   causallm_android_api.h
 * @date   22 January 2025
 * @brief  CausalLM Android C API (No JNI required)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Android Build Support
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_ANDROID_API_H__
#define __CAUSALLM_ANDROID_API_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ANDROID
#include <android/log.h>
#define CAUSALLM_LOG_TAG "CausalLM_Android"
#define CAUSALLM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, CAUSALLM_LOG_TAG, __VA_ARGS__)
#define CAUSALLM_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, CAUSALLM_LOG_TAG, __VA_ARGS__)
#else
#define CAUSALLM_LOGI(...) printf(__VA_ARGS__)
#define CAUSALLM_LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

/**
 * @brief CausalLM handle type
 */
typedef void* causallm_handle_t;

/**
 * @brief CausalLM return codes
 */
typedef enum {
    CAUSALLM_ERROR_NONE = 0,
    CAUSALLM_ERROR_INVALID_PARAMETER = -1,
    CAUSALLM_ERROR_OUT_OF_MEMORY = -2,
    CAUSALLM_ERROR_INVALID_OPERATION = -3,
    CAUSALLM_ERROR_NOT_SUPPORTED = -4,
    CAUSALLM_ERROR_UNKNOWN = -999
} causallm_error_e;

/**
 * @brief Create CausalLM instance
 * @param[out] handle CausalLM handle to be created
 * @return CAUSALLM_ERROR_NONE on success, otherwise error code
 */
causallm_error_e causallm_create(causallm_handle_t *handle);

/**
 * @brief Load model from file path
 * @param[in] handle CausalLM handle
 * @param[in] model_path Path to the model configuration directory
 * @return CAUSALLM_ERROR_NONE on success, otherwise error code
 */
causallm_error_e causallm_load_model(causallm_handle_t handle, const char *model_path);

/**
 * @brief Run inference with input text
 * @param[in] handle CausalLM handle
 * @param[in] input_text Input text for inference
 * @param[out] output_text Generated output text (caller must free)
 * @return CAUSALLM_ERROR_NONE on success, otherwise error code
 */
causallm_error_e causallm_inference(causallm_handle_t handle, 
                                   const char *input_text, 
                                   char **output_text);

/**
 * @brief Set inference parameters
 * @param[in] handle CausalLM handle
 * @param[in] max_tokens Maximum tokens to generate
 * @param[in] temperature Temperature for sampling
 * @param[in] top_p Top-p value for nucleus sampling
 * @return CAUSALLM_ERROR_NONE on success, otherwise error code
 */
causallm_error_e causallm_set_params(causallm_handle_t handle, 
                                     int max_tokens, 
                                     float temperature, 
                                     float top_p);

/**
 * @brief Free output text allocated by causallm_inference
 * @param[in] output_text Text to be freed
 */
void causallm_free_text(char *output_text);

/**
 * @brief Destroy CausalLM instance
 * @param[in] handle CausalLM handle to be destroyed
 * @return CAUSALLM_ERROR_NONE on success, otherwise error code
 */
causallm_error_e causallm_destroy(causallm_handle_t handle);

/**
 * @brief Get error message string
 * @param[in] error_code Error code
 * @return Error message string
 */
const char* causallm_get_error_message(causallm_error_e error_code);

#ifdef __cplusplus
}
#endif

#endif /* __CAUSALLM_ANDROID_API_H__ */