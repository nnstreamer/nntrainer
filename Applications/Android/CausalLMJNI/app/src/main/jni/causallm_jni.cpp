// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   causallm_jni.cpp
 * @date   01 January 2025
 * @brief  JNI implementation for CausalLM Android application
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Generated for CausalLM Android build
 * @bug    No known bugs except for NYI items
 */

#include "causallm_jni.h"
#include <android/log.h>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>

// Include CausalLM headers (these would be from the PR #3344)
// Note: These includes assume the CausalLM implementation is available
#include <causal_lm.h>
#include <factory.h>
#include <json.hpp>

#define LOG_TAG "CausalLMJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using json = nlohmann::json;

/**
 * @brief Load JSON file helper function
 */
json LoadJsonFile(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    
    json data;
    file >> data;
    return data;
}

JNIEXPORT jlong JNICALL
Java_com_applications_causallmjni_MainActivity_createCausalLMModel(
    JNIEnv *env, jobject thiz, jstring config_path) {
    
    try {
        // Convert jstring to std::string
        const char *config_path_str = env->GetStringUTFChars(config_path, 0);
        std::string model_path(config_path_str);
        env->ReleaseStringUTFChars(config_path, config_path_str);
        
        LOGI("Creating CausalLM model from path: %s", model_path.c_str());
        
        // Load configuration files
        json cfg = LoadJsonFile(model_path + "/config.json");
        json generation_cfg = LoadJsonFile(model_path + "/generation_config.json");
        json nntr_cfg = LoadJsonFile(model_path + "/nntr_config.json");
        
        // Register CausalLM models to factory
        causallm::Factory::Instance().registerModel(
            "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::CausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        causallm::Factory::Instance().registerModel(
            "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        causallm::Factory::Instance().registerModel(
            "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        
        // Create model instance
        auto model = causallm::Factory::Instance().create(
            cfg["architectures"].get<std::vector<std::string>>()[0], 
            cfg, generation_cfg, nntr_cfg);
        
        if (!model) {
            LOGE("Failed to create CausalLM model");
            return 0;
        }
        
        LOGI("CausalLM model created successfully");
        return reinterpret_cast<jlong>(model.release());
        
    } catch (const std::exception &e) {
        LOGE("Exception in createCausalLMModel: %s", e.what());
        return 0;
    }
}

JNIEXPORT jint JNICALL
Java_com_applications_causallmjni_MainActivity_initializeModel(
    JNIEnv *env, jobject thiz, jlong model_pointer) {
    
    try {
        if (model_pointer == 0) {
            LOGE("Invalid model pointer");
            return -1;
        }
        
        causallm::CausalLM *model = reinterpret_cast<causallm::CausalLM*>(model_pointer);
        
        LOGI("Initializing CausalLM model");
        model->initialize();
        
        LOGI("CausalLM model initialized successfully");
        return 0;
        
    } catch (const std::exception &e) {
        LOGE("Exception in initializeModel: %s", e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL
Java_com_applications_causallmjni_MainActivity_loadWeights(
    JNIEnv *env, jobject thiz, jlong model_pointer, jstring weight_path) {
    
    try {
        if (model_pointer == 0) {
            LOGE("Invalid model pointer");
            return -1;
        }
        
        // Convert jstring to std::string
        const char *weight_path_str = env->GetStringUTFChars(weight_path, 0);
        std::string weights_path(weight_path_str);
        env->ReleaseStringUTFChars(weight_path, weight_path_str);
        
        causallm::CausalLM *model = reinterpret_cast<causallm::CausalLM*>(model_pointer);
        
        LOGI("Loading weights from: %s", weights_path.c_str());
        model->load_weight(weights_path);
        
        LOGI("Weights loaded successfully");
        return 0;
        
    } catch (const std::exception &e) {
        LOGE("Exception in loadWeights: %s", e.what());
        return -1;
    }
}

JNIEXPORT jstring JNICALL
Java_com_applications_causallmjni_MainActivity_runInference(
    JNIEnv *env, jobject thiz, jlong model_pointer, jstring input_text, jboolean do_sample) {
    
    try {
        if (model_pointer == 0) {
            LOGE("Invalid model pointer");
            return env->NewStringUTF("");
        }
        
        // Convert jstring to std::string
        const char *input_str = env->GetStringUTFChars(input_text, 0);
        std::string prompt(input_str);
        env->ReleaseStringUTFChars(input_text, input_str);
        
        causallm::CausalLM *model = reinterpret_cast<causallm::CausalLM*>(model_pointer);
        
        LOGI("Running inference with prompt: %s", prompt.c_str());
        
        // Note: The original CausalLM::run method prints to stdout
        // For Android, we might need to modify it to return the result
        // For now, we'll use a placeholder implementation
        model->run(prompt, static_cast<bool>(do_sample));
        
        // TODO: Modify CausalLM to return generated text instead of printing
        // For now, return a placeholder
        std::string result = "Generated text would appear here";
        
        LOGI("Inference completed");
        return env->NewStringUTF(result.c_str());
        
    } catch (const std::exception &e) {
        LOGE("Exception in runInference: %s", e.what());
        return env->NewStringUTF("");
    }
}

JNIEXPORT void JNICALL
Java_com_applications_causallmjni_MainActivity_destroyModel(
    JNIEnv *env, jobject thiz, jlong model_pointer) {
    
    try {
        if (model_pointer != 0) {
            causallm::CausalLM *model = reinterpret_cast<causallm::CausalLM*>(model_pointer);
            delete model;
            LOGI("CausalLM model destroyed");
        }
    } catch (const std::exception &e) {
        LOGE("Exception in destroyModel: %s", e.what());
    }
}