// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move picogpt model creating to separate sourcefile
 * @brief  task runner for the picogpt
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#if defined(ENABLE_TEST)
#include <gtest/gtest.h>
#endif

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#ifdef PROFILE
#include <profiler.h>
#endif

#include <android/log.h>
#define ANDROID_LOG_E(fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "nntrainer", fmt, ##__VA_ARGS__)
#define ANDROID_LOG_I(fmt, ...) \
  __android_log_print(ANDROID_LOG_INFO, "nntrainer", fmt, ##__VA_ARGS__)
#define ANDROID_LOG_D(fmt, ...) \
  __android_log_print(ANDROID_LOG_DEBUG, "nntrainer", fmt, ##__VA_ARGS__)

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value);

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value);

/**
 * @brief create Picogpt18 Model
 *
 * @param input shape input dimension of model
 * @return Model return model
 */
ml::train::Model *createPicogpt();

/**
 * @brief display iteration status
 *
 * @param count iteration count
 * @param loss loss value
 * @param batch size batch size
 * @return string display string
 */
std::string displayProgress(const int count, float loss, int batch_size);


/**
 * @brief this is main for android jni interface to train
 *
 * @param argc number of argument
 * @param argv argment list
 * @param model model pointer
 */
int init(int argc, char *argv[], ml::train::Model *model);

std::string inferModel(std::string path, std::string sentence,
                       ml::train::Model *model_);

/**
 * @brief check if the model is properly destoryed
 * @return true for destoryed.
 *
 */
bool modelDestroyed();

std::string getInferResult();
