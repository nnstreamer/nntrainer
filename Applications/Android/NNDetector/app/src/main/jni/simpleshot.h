// SPDX-License-Identifier: Apache-2.0
/**
 * @file   simpleshot.h
 * @date   24 Oct 2023
 * @brief  image recognition/detection modules
 * @author HS.Kim <hs0207.kim@samsung.com>
 * @bug    No known bugs
 */
#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(ENABLE_TEST)
#include <gtest/gtest.h>
#endif

#include <app_context.h>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
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
using DatasetHandle = std::shared_ptr<ml::train::Dataset>;
using UserDataType = std::unique_ptr<nntrainer::simpleshot::DirDataLoader>;

/**
 * @brief this is main for android jni interface to initialize the detection
 * model
 *
 * @param argc number of argument
 * @param argv argument list
 * @return model pointer
 */
ml::train::Model *initialize_det(int argc, char *argv[]);

/**
 * @brief this is main for android jni interface to initialize the recognition
 * model
 *
 * @param argc number of argument
 * @param argv argument list
 * @return model pointer
 */
ml::train::Model *initialize_rec(int argc, char *argv[]);

/**
 * @brief main function to train
 *
 * @param argc number of argument
 * @param argv argument list
 * @param model model pointer
 */
void train_prototypes(int argc, char *argv[], ml::train::Model *det_model_,
                      ml::train::Model *rec_model_);

/**
 * @brief main function to test
 *
 * @param argc number of argument
 * @param argv argument list
 * @param det_model detection model pointer
 * @param rec_model recognition model pointer
 * @return string test result
 */
std::string test_prototypes(int argc, char *argv[],
                            ml::train::Model *det_model_,
                            ml::train::Model *rec_model_);

/**
 * @brief main function to test
 *
 * @param argc number of argument
 * @param argv argument list
 * @param det_model detection model pointer
 * @return string detection result
 */
std::string run_detector(int argc, char *argv[], ml::train::Model *det_model_);
