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

#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <dataset.h>
#include <tensor_dim.h>

namespace nntrainer {

/**
 * @brief     Aliasing from ccapi ml::train
 */
using DatasetType = ml::train::DatasetType;
using DatasetDataUsageType = ml::train::DatasetDataUsageType;
using datagen_cb = ml::train::datagen_cb;

/**
 * @class   DataBuffer Data Buffers
 * @brief   Data Buffer for read and manage data
 */
class DataBuffer : public ml::train::Dataset {
public:
  /**
   * @brief     Create Buffer
   * @retval    DataBuffer
   */
  DataBuffer(DatasetType type);

  /**
   * @brief     Initialize Buffer with data buffer private variables
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int init();

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  virtual void updateData(DatasetDataUsageType type) = 0;

  /**
   * @brief     function for thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int run(DatasetDataUsageType type);

  /**
   * @brief     clear thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int clear(DatasetDataUsageType type);

  /**
   * @brief     clear all thread ( training, validation, test )
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int clear();

  /**
   * @brief     get Data from Data Buffer using databuffer param
   * @param[in] BufferType training, validation, test
   * @param[out] out feature data ( batch_size size ), a contiguous and
   * allocated memory block should be passed
   * @param[out] label label data ( batch_size size ), a contiguous and
   * allocated memory block should be passed
   * @retval    true/false
   */
  bool getDataFromBuffer(DatasetDataUsageType type, float *out, float *label);

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
   * @brief     set batch size
   * @param[in] n batch size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setBatchSize(unsigned int n);

  /**
   * @brief     set feature size
   * @param[in] feature batch size. It is equal to input layer's hidden size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
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
   * @param[in] count calculated set ( batch_size size )
   * @param[in] type buffer type ( DATA_TRAIN, DATA_VAL, DATA_TEST )
   * @retval void
   */
  void displayProgress(const int count, DatasetDataUsageType type, float loss);

  /**
   * @brief     return validation of data set
   * @retval validation
   */
  bool *getValidation() { return validation; }

  /**
   * @brief     set property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     set property to allow setting user_data for cb
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<void *> values);

  /**
   * @brief     set function pointer for each type
   * @param[in] type Buffer Type
   * @param[in] call back function pointer
   * @param[in] user_data user_data of the callback
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setGeneratorFunc(DatasetDataUsageType type, datagen_cb func,
                               void *user_data = nullptr);

  /**
   * @brief     set train data file name
   * @param[in] type data type : DATA_TRAIN, DATA_VAL, DATA_TEST
   * @param[in] path file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setDataFile(DatasetDataUsageType type, std::string path) {
    return setDataFile(type, path);
  }

  /**
   * @brief property type of databuffer
   *
   */
  enum class PropertyType {
    train_data = 0,
    val_data = 1,
    test_data = 2,
    buffer_size = 3,
    unknown = 4
  };

protected:
  /**
   * @brief Number of Data Set
   */
  static constexpr const unsigned int NBUFTYPE = 4;

  /**
   * @brief state of the data buffer while getting the data
   */
  typedef enum {
    DATA_NOT_READY = 0,
    DATA_READY = 1,
    DATA_END = 2,
    DATA_ERROR = 3,
  } DataStatus;

  /**
   * @brief     status of thread
   */
  DataStatus trainReadyFlag;
  DataStatus valReadyFlag;
  DataStatus testReadyFlag;

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
   * @brief     batch size
   */
  unsigned int batch_size;

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

  /**
   * @brief     The type of data buffer
   */
  DatasetType data_buffer_type;

  /** The user_data to be used for the data generator callback */
  void *user_data;

  /**
   * @brief     set property
   * @param[in] type type of property
   * @param[in] value string value of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(const DataBuffer::PropertyType type,
                          std::string &value);
};

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_H__ */
