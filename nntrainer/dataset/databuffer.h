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
   * @brief   Create DataBuffer with a producer
   *
   */
  DataBuffer(std::unique_ptr<DataProducer> &&producer_);

  /**
   * @brief Destroy the Data Buffer object
   *
   */
  ~DataBuffer();

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
   * @brief     Display Progress
   * @param[in] count calculated set ( batch_size size )
   * @param[in] type buffer type ( DATA_TRAIN, DATA_VAL, DATA_TEST )
   * @retval void
   */
  void displayProgress(const int count, float loss);

  /**
   * @brief     set property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief     set property to allow setting user_data for cb
   * @todo   deprecate
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<void *> values);

protected:
  std::shared_ptr<DataProducer> producer;
  std::weak_ptr<BatchQueue> bq_view;
  using Props = std::tuple<PropsBufferSize>;
  std::unique_ptr<Props> db_props;

  /// @todo this must be handled from the capi side. favorably, deprecate
  /// "user_data", callback
  /** The user_data to be used for the data generator callback */
  void *user_data;
};

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_H__ */
