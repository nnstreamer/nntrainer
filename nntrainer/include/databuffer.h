/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	databuffer.h
 * @date	04 December 2019
 * @brief	This is buffer object to handle big data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __DATABUFFER_H__
#define __DATABUFFER_H__
#ifdef __cplusplus

#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <tensor_dim.h>
#include <thread>
#include <util_func.h>
#include <vector>

/*
 * @brief Number of Data Set
 */
#define NBUFTYPE 4

typedef std::vector<std::vector<std::vector<std::vector<float>>>> vec_4d;

#define SET_VALIDATION(val)                                              \
  do {                                                                   \
    for (DataType i = DATA_TRAIN; i < DATA_UNKNOWN; i = DataType(i + 1)) \
      validation[i] = val;                                               \
  } while (0)

#define NN_EXCEPTION_NOTI(val)                             \
  do {                                                     \
    switch (type) {                                        \
    case BUF_TRAIN: {                                      \
      std::lock_guard<std::mutex> lgtrain(readyTrainData); \
      trainReadyFlag = val;                                \
      cv_train.notify_all();                               \
    } break;                                               \
    case BUF_VAL: {                                        \
      std::lock_guard<std::mutex> lgval(readyValData);     \
      valReadyFlag = val;                                  \
      cv_val.notify_all();                                 \
    } break;                                               \
    case BUF_TEST: {                                       \
      std::lock_guard<std::mutex> lgtest(readyTestData);   \
      testReadyFlag = val;                                 \
      cv_test.notify_all();                                \
    } break;                                               \
    default:                                               \
      break;                                               \
    }                                                      \
  } while (0)

typedef enum {
  DATA_NOT_READY = 0,
  DATA_READY = 1,
  DATA_END = 2,
  DATA_ERROR = 3,
} DataStatus;

namespace nntrainer {

/**
 * @brief     Enumeration of buffer type
 *            0. BUF_TRAIN ( Buffer for training )
 *            1. BUF_VAL ( Buffer for validation )
 *            2. BUF_TEST ( Buffer for test )
 *            3. BUF_UNKNOWN
 */
typedef enum { BUF_TRAIN, BUF_VAL, BUF_TEST, BUF_UNKNOWN } BufferType;

/**
 * @brief     Enumeration of data type
 *            0. DATA_TRAIN ( Data for training )
 *            1. DATA_VAL ( Data for validation )
 *            2. DATA_TEST ( Data for test )
 *            3. DATA_LABEL ( Data for test )
 *            4. DATA_UNKNOWN
 */
typedef enum {
  DATA_TRAIN,
  DATA_VAL,
  DATA_TEST,
  DATA_LABEL,
  DATA_UNKNOWN
} DataType;

/**
 * @class   DataBuffer Data Buffers
 * @brief   Data Buffer for read and manage data
 */
class DataBuffer {
public:
  /**
   * @brief     Create Buffer
   * @retval    DataBuffer
   */
  DataBuffer() :
    train_running(),
    val_running(),
    test_running(),
    train_thread(),
    val_thread(),
    test_thread() {
    SET_VALIDATION(false);
    class_num = 0;
    cur_train_bufsize = 0;
    cur_val_bufsize = 0;
    cur_test_bufsize = 0;
    train_bufsize = 0;
    val_bufsize = 0;
    test_bufsize = 0;
    max_train = 0;
    max_val = 0;
    max_test = 0;
    rest_train = 0;
    rest_val = 0;
    rest_test = 0;
    mini_batch = 0;
    train_running = false;
    val_running = false;
    test_running = false;
    rng.seed(getSeed());
  };

  /**
   * @brief     Destructor
   */
  virtual ~DataBuffer(){};

  /**
   * @brief     Initialize Buffer with data buffer private variables
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int init() = 0;

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  virtual void updateData(BufferType type) = 0;

  /**
   * @brief     function for thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int run(BufferType type);

  /**
   * @brief     clear thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int clear(BufferType type);

  /**
   * @brief     clear all thread ( training, validation, test )
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int clear();

  /**
   * @brief     get Data from Data Buffer using databuffer param
   * @param[in] BufferType training, validation, test
   * @param[in] outVec feature data ( minibatch size )
   * @param[in] outLabel label data ( minibatch size )
   * @retval    true/false
   */
  virtual bool getDataFromBuffer(
    BufferType type,
    std::vector<std::vector<std::vector<std::vector<float>>>> &out_vec,
    std::vector<std::vector<std::vector<std::vector<float>>>> &out_label);

  /**
   * @brief     set number of class
   * @param[in] number of class
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setClassNum(unsigned int n);

  /**
   * @brief     set buffer size
   * @param[in] buffer size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setBufSize(unsigned int n);

  /**
   * @brief     set mini batch size
   * @param[in] mini batch size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setMiniBatch(unsigned int n);

  /**
   * @brief     set feature size
   * @param[in] feature batch size. It is equal to input layer's hidden size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  /* virtual int setFeatureSize(unsigned int n); */
  virtual int setFeatureSize(TensorDim indim);

  /**
   * @brief     set feature size
   * @retval max_train
   */
  unsigned int getMaxTrain() { return max_train; }

  /**
   * @brief     set feature size
   * @retval max_val
   */
  unsigned int getMaxVal() { return max_val; }

  /**
   * @brief     set feature size
   * @retval max_test
   */
  unsigned int getMaxTest() { return max_test; }

  /**
   * @brief     Display Progress
   * @param[in] count calculated set ( mini_batch size )
   * @param[in] type buffer type ( BUF_TRAIN, BUF_VAL, BUF_TEST )
   * @retval void
   */
  void displayProgress(const int count, BufferType type, float loss);

  /**
   * @brief     return validation of data set
   * @retval validation
   */
  bool *getValidation() { return validation; }

  /**
   * @brief     status of thread
   */
  DataStatus trainReadyFlag;
  DataStatus valReadyFlag;
  DataStatus testReadyFlag;

protected:
  /**
   * @brief     Data Queues for each data set
   */
  std::vector<std::vector<float>> train_data;
  std::vector<std::vector<float>> train_data_label;
  std::vector<std::vector<float>> val_data;
  std::vector<std::vector<float>> val_data_label;
  std::vector<std::vector<float>> test_data;
  std::vector<std::vector<float>> test_data_label;

  /**
   * @brief     feature size
   */
  TensorDim input_dim;

  /**
   * @brief     number of class
   */
  unsigned int class_num;

  /**
   * @brief     number of remain data for each data queue
   */
  unsigned int cur_train_bufsize;
  unsigned int cur_val_bufsize;
  unsigned int cur_test_bufsize;

  /**
   * @brief     queue size for each data set
   */
  unsigned int train_bufsize;
  unsigned int val_bufsize;
  unsigned int test_bufsize;

  unsigned int max_train;
  unsigned int max_val;
  unsigned int max_test;

  /**
   * @brief     remain data set size
   */
  unsigned int rest_train;
  unsigned int rest_val;
  unsigned int rest_test;

  /**
   * @brief     mini batch size
   */
  unsigned int mini_batch;

  /**
   * @brief     flags to check status
   */
  bool train_running;
  bool val_running;
  bool test_running;

  /**
   * @brief     ids to check duplication
   */
  std::vector<unsigned int> train_mark;
  std::vector<unsigned int> val_mark;
  std::vector<unsigned int> test_mark;

  /**
   * @brief     threads to generate data for queue
   */
  std::thread train_thread;
  std::thread val_thread;
  std::thread test_thread;

  std::vector<std::string> labels;
  bool validation[NBUFTYPE];

  /**
   * @brief     return random int value between min to max
   * @param[in] min minimum vaule
   * @param[in] max maximum value
   * @retval    int return value
   */
  int rangeRandom(int min, int max);

  std::mt19937 rng;
};

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_H__ */
