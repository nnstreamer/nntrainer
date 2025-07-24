// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet
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

#include <dataloader.h>

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

using UserDataType = std::unique_ptr<nntrainer::resnet::DataLoader>;
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
std::vector<LayerHandle> createResnet18Graph(std::string input_shape,
                                             unsigned int unit);

/**
 * @brief create Resnet18 Model
 *
 * @param input shape input dimension of model
 * @return Model return model
 */
ml::train::Model *createResnet18(std::string input_shape, unsigned int unit);

/**
 * @brief callback to get the one training data
 *
 * @param input 1d buffer for input data
 * @param label one hot vector
 * @param last  true / false : indicate the end of dataset
 * @param user_data  user data for callback
 * @return status int
 */
int trainData_cb(float **input, float **label, bool *last, void *user_data);

/**
 * @brief callback to get the one validation data
 *
 * @param input 1d buffer for input data
 * @param label one hot vector
 * @param last  true / false : indicate the end of dataset
 * @param user_data  user data for callback
 * @return status int
 */
int validData_cb(float **input, float **label, bool *last, void *user_data);

/**
 * @brief main function to train
 *
 * @param epochs number of epochs
 * @param batch size batch size
 * @param train_user_data userdata for train
 * @param train_user_data userdata for validataion
 * @param model  model
 */
/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data, UserDataType &valid_user_data,
                  ml::train::Model *model);

/**
 * @brief Fake Data buffer generator
 *
 * @param batch size batch size
 * @simulated_data_size total number of data
 * @data_split split ratio
 * @return two UserDataType ( one for Training , and one for Validation )
 */
std::array<UserDataType, 2>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int simulated_data_size,
                        unsigned int data_split);

/**
 * @brief Data buffer generator in Directory
 *
 * @param string directory path , within this, train and test directory should
 * exist.
 * @param split_ratio split ratio (validation / total traing data)
 * @param label_len length of label (it usally equals with num class)
 * @param channel channel
 * @param width width
 * @param height height
 * @param is train true for training, false for validation
 * @return two UserDataType ( one for Training , and one for Validation )
 */
std::array<UserDataType, 2>
createDirDataGenerator(const std::string dir, float split_ratio,
                       unsigned int label_len, unsigned int channel,
                       unsigned int width, unsigned int height, bool is_train);
/**
 * @brief this is main for android jni interface to train
 *
 * @param argc number of argument
 * @param argv argment list
 * @param model model pointer
 */
int init(int argc, char *argv[], ml::train::Model *model);

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
 * @brief set the stop (training & testing)
 *
 */
void setStop();

/**
 * @brief inference model using test dataset
 * @param argc num of argv
 * @param argv string argument to run
 * @param model_ model
 * @return string test result
 *
 */
std::string testModel(int argc, char *argv[], ml::train::Model *model_);

/**
 * @brief inference model
 * @param argc num of argv
 * @param argv string argument to run
 * @param model_ model
 * @return string test result
 *
 */
std::string inferModel(int argc, char *argv[], uint8_t *pBmp,
                       ml::train::Model *model_);

/**
 * @brief getter for testing status
 * @return string test information
 *
 */
std::string getTestingStatus();

/**
 * @brief check if the model is properly destoryed
 * @return true for destoryed.
 *
 */
bool modelDestroyed();
