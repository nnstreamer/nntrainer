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
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <batch_queue.h>
#include <data_producers.h>
#include <dataset.h>
#include <tensor_dim.h>

namespace nntrainer {

/**
 * @brief     Aliasing from ccapi ml::train
 */
using DatasetType = ml::train::DatasetType;
using datagen_cb = ml::train::datagen_cb;
using TensorDim = ml::train::TensorDim;

class PropsBufferSize;

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
   * @brief   Create DataBuffer with a producer
   *
   */
  DataBuffer(std::unique_ptr<DataProducer> &&producer_);

  /**
   * @brief Destroy the Data Buffer object
   *
   */
  virtual ~DataBuffer();

  /**
   * @brief     Initialize Buffer with data buffer private variables
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int init();

  /**
   * @brief prepare iteration a head of time with a dedicated worker. The
   * iteration prepared can be retrieved with @a fetch();
   * @remark the batch dimension of input_dims / label_dims must be same for
   * all.
   * @param input_dims dimension of input_dims
   * @param label_dims dimension of label_dims
   * @return std::future<std::shared_ptr<BatchQueue>> Buffer Queue object,
   * release this pointer after calling @a fetch() is done to invalidate
   * subsequent call of @a fetch()
   */
  std::future<std::shared_ptr<BatchQueue>>
  startFetchWorker(const std::vector<TensorDim> &input_dims,
                   const std::vector<TensorDim> &label_dims);

  /**
   * @brief Get the Iteration object
   * @note  the first element of returned Iteration denotes whether current
   * epoch has ended.
   *
   * @throw std::invalid_argument if @a startFetchWorker hasn't been called or
   * the return value of startFetchWorker has been invalidated.
   * @return std::unique_ptr<DataProducer::Iteration> iteration
   */
  std::unique_ptr<DataProducer::Iteration> fetch();

  /**
   * @brief Get the Generator object and the generator object returns a batch
   * upon call
   * @remark the batch dimension of input_dims / label_dims must be same for
   * all.
   *
   * @param input_dims dimension of input_dims
   * @param label_dims dimension of label_dims
   * @return DataProducer::Generator which generates an iteration
   */
  DataProducer::Generator batcher(const std::vector<TensorDim> &input_dims,
                                  const std::vector<TensorDim> &label_dims);

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @retval    void
   */
  virtual void updateData(){};

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
  void setProperty(const std::vector<std::string> &values) override;

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

  /******************* v2 members ********************/
  std::shared_ptr<DataProducer> producer;
  std::weak_ptr<BatchQueue> bq_view;
  using Props = std::tuple<PropsBufferSize>;
  std::unique_ptr<Props> db_props;

  /** The user_data to be used for the data generator callback */
  void *user_data;
};

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_H__ */
