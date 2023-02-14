// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
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

#include <cifar_dataloader.h>

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

using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

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

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value);

/**
 * @brief resnet block
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param filters number of filters
 * @param kernel_size number of kernel_size
 * @param downsample downsample to make output size 0
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> resnetBlock(const std::string &block_name,
                                     const std::string &input_name, int filters,
                                     int kernel_size, bool downsample);

/**
 * @brief Create resnet 18
 *
 * @return vector of layers that contain full graph of resnet18
 */
std::vector<LayerHandle> createResnet18Graph();

/// @todo update createResnet18 to be more generic
ModelHandle createResnet18();

int trainData_cb(float **input, float **label, bool *last, void *user_data);

int validData_cb(float **input, float **label, bool *last, void *user_data);

/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data, UserDataType &valid_user_data);

std::array<UserDataType, 2>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int simulated_data_size,
                        unsigned int data_split);

std::array<UserDataType, 2>
createRealDataGenerator(const std::string &directory, unsigned int batch_size,
                        unsigned int data_split);

int init(int argc, char *argv[]);
