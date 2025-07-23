// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file causallm_jni.cpp
 * @date 25 January 2025
 * @brief JNI wrapper for CausalLM application
 * @author Samsung Electronics Co., Ltd.
 * @bug No known bugs except for NYI items
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <memory>
#include <fstream>

#include <causal_lm.h>
#include <qwen3_causallm.h>
#include <qwen3_moe_causallm.h>
#include <factory.h>

#define LOG_TAG "CausalLM_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace causallm;

static std::unique_ptr<CausalLM> g_causallm_model = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_ai_nnstreamer_nntrainer_causallm_CausalLMActivity_initializeModel(
    JNIEnv *env, jobject thiz, jstring model_path) {
    
    const char *model_path_str = env->GetStringUTFChars(model_path, nullptr);
    if (model_path_str == nullptr) {
        LOGE("Failed to get model path string");
        return JNI_FALSE;
    }
    
    try {
        std::string model_dir(model_path_str);
        
        // Load configuration files
        std::string config_path = model_dir + "/config.json";
        std::string generation_config_path = model_dir + "/generation_config.json";
        std::string nntr_config_path = model_dir + "/nntr_config.json";
        
        json config = LoadJsonFile(config_path);
        json generation_config = LoadJsonFile(generation_config_path);
        json nntr_config = LoadJsonFile(nntr_config_path);
        
        // Determine model architecture
        std::string architecture = config["architectures"][0];
        
        if (architecture == "Qwen3ForCausalLM") {
            g_causallm_model = std::make_unique<Qwen3CausalLM>(config, generation_config, nntr_config);
        } else if (architecture == "Qwen3MoeForCausalLM") {
            g_causallm_model = std::make_unique<Qwen3MoECausalLM>(config, generation_config, nntr_config);
        } else {
            // Default to base CausalLM
            g_causallm_model = std::make_unique<CausalLM>(config, generation_config, nntr_config);
        }
        
        g_causallm_model->initialize();
        
        // Load weights
        std::string weight_path = model_dir + "/" + nntr_config["model_file_name"].get<std::string>();
        g_causallm_model->load_weight(weight_path);
        
        LOGI("CausalLM model initialized successfully");
        
    } catch (const std::exception &e) {
        LOGE("Failed to initialize model: %s", e.what());
        env->ReleaseStringUTFChars(model_path, model_path_str);
        return JNI_FALSE;
    }
    
    env->ReleaseStringUTFChars(model_path, model_path_str);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_ai_nnstreamer_nntrainer_causallm_CausalLMActivity_runInference(
    JNIEnv *env, jobject thiz, jstring input_text, jboolean do_sample) {
    
    if (g_causallm_model == nullptr) {
        LOGE("Model not initialized");
        return env->NewStringUTF("Error: Model not initialized");
    }
    
    const char *input_str = env->GetStringUTFChars(input_text, nullptr);
    if (input_str == nullptr) {
        LOGE("Failed to get input text string");
        return env->NewStringUTF("Error: Invalid input text");
    }
    
    try {
        std::string prompt(input_str);
        g_causallm_model->run(prompt, static_cast<bool>(do_sample));
        
        // Get the output (assuming single batch)
        std::string output = g_causallm_model->output_list[0];
        
        env->ReleaseStringUTFChars(input_text, input_str);
        return env->NewStringUTF(output.c_str());
        
    } catch (const std::exception &e) {
        LOGE("Failed to run inference: %s", e.what());
        env->ReleaseStringUTFChars(input_text, input_str);
        return env->NewStringUTF(("Error: " + std::string(e.what())).c_str());
    }
}

JNIEXPORT void JNICALL
Java_ai_nnstreamer_nntrainer_causallm_CausalLMActivity_destroyModel(
    JNIEnv *env, jobject thiz) {
    
    if (g_causallm_model != nullptr) {
        g_causallm_model.reset();
        LOGI("CausalLM model destroyed");
    }
}

JNIEXPORT jboolean JNICALL
Java_ai_nnstreamer_nntrainer_causallm_CausalLMActivity_isModelInitialized(
    JNIEnv *env, jobject thiz) {
    
    return (g_causallm_model != nullptr) ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"