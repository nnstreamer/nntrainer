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

#include <condition_variable>
#include <memory>
#include <mutex>
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
using datagen_cb = ml::train::datagen_cb;
using TensorDim = ml::train::TensorDim;

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
   * @retval    void
   */
  virtual void updateData() = 0;

  /**
   * @brief     function for thread ( training, validation, test )
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int run();

  /**
   * @brief     clear thread ( training, validation, test )
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int clear();

  /**
   * @brief     get Data from Data Buffer using databuffer param
   * @param[out] out feature data ( batch_size size ), a contiguous and
   * allocated memory block should be passed
   * @param[out] label label data ( batch_size size ), a contiguous and
   * allocated memory block should be passed
   * @retval    true/false
   */
  bool getDataFromBuffer(float *out, float *label);

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
  virtual int setFeatureSize(const TensorDim &indim);

  /**
   * @brief     Display Progress
   * @param[in] count calculated set ( batch_size size )
   * @param[in] type buffer type ( DATA_TRAIN, DATA_VAL, DATA_TEST )
   * @retval void
   */
  void displayProgress(const int count, float loss);

  /**
   * @brief     return validation of data set
   * @retval validation
   */
  bool isValid() { return initialized; }

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
   * @param[in] call back function pointer
   * @param[in] user_data user_data of the callback
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setGeneratorFunc(datagen_cb func, void *user_data = nullptr);

  /**
   * @brief     set train data file name
   * @param[in] path file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setDataFile(const std::string &path);

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
   * @brief state of the data buffer while getting the data
   */
  enum class DataStatus {
    DATA_NOT_READY = 0,
    DATA_READY = 1,
    DATA_END = 2,
    DATA_ERROR = 3,
  };

  /**
   * @brief Set the State And Notify to the condition variable waiting for it
   *
   * @param status status to change
   */
  void setStateAndNotify(const DataStatus status);

  DatasetType data_buffer_type; /**< data buffer type */

  /** data queue and producer/consumer status related variables */
  std::thread
    batch_producer;        /** thread generates a single batches to the queue */
  std::mutex data_lock;    /**< data queue mutex */
  std::mutex status_mutex; /**< data status mutex */
  std::condition_variable cv_status; /**< condition variable for status */
  std::vector<std::vector<float>> data_q;
  std::vector<std::vector<float>> label_q;
  DataStatus queue_status;  /**< status of the queue */
  unsigned int buf_size;    /**< queue size */
  unsigned int cur_bufsize; /**<  number of data in the data queue */

  TensorDim input_dim;     /**< feature size */
  unsigned int class_num;  /**< number of class */
  unsigned int batch_size; /**< batch size */

  /**< @todo below variable should be owned by databuffer that has fixed size */
  unsigned int samples_per_epoch; /**< size of samples in the dataset */
  unsigned int remaining_samples_per_epoch; /**< size of samples remaining in
                                               current epoch */

  bool is_running;  /**< flag to check if running */
  bool initialized; /**< check if current buffer is in valid state */

  /**
   * @brief     return random int value between min to max
   * @param[in] min minimum vaule
   * @param[in] max maximum value
   * @retval    int return value
   */
  int rangeRandom(int min, int max);

  std::mt19937 rng;

  std::exception_ptr consumer_exception_ptr; /**< exception ptr for consumer to
                                                catch when producer is dead */

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
