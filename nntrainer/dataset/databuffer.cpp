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
 * @file	databuffer.cpp
 * @date	04 December 2019
 * @brief	This is buffer object to handle big data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <base_properties.h>
#include <cassert>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <databuffer.h>
#include <functional>
#include <iomanip>
#include <mutex>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <parse_util.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief Props containing buffer size value
 *
 */
class PropsBufferSize : public Property<unsigned int> {
public:
  /**
   * @brief Construct a new props min object with a default value
   *
   * @param value default value
   */
  PropsBufferSize(unsigned int value = 1) { set(value); }
  bool isValid(const unsigned int &v) { return v > 0; }
  static constexpr const char *key = "buffer_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

constexpr char USER_DATA[] = "user_data";

DataBuffer::DataBuffer(std::unique_ptr<DataProducer> &&producer_) :
  producer(std::move(producer_)),
  db_props(new Props()) {}

DataBuffer::~DataBuffer(){};

std::future<std::shared_ptr<BatchQueue>>
DataBuffer::startFetchWorker(const std::vector<TensorDim> &input_dims,
                             const std::vector<TensorDim> &label_dims) {
  NNTR_THROW_IF(!producer, std::invalid_argument) << "producer does not exist";
  auto bq = std::make_shared<BatchQueue>(std::get<PropsBufferSize>(*db_props));
  auto generator = producer->finalize(input_dims, label_dims);
  bq_view = bq;

  return std::async(std::launch::async, [bq, generator] {
    while (true) {
      try {
        /// @note add dimension check in debug mode
        auto iteration = generator();
        auto last = std::get<0>(iteration);
        bq->wait_and_push(std::move(iteration));
        if (last == true) {
          break;
        }
      } catch (std::exception &e) {
        bq->wait_and_push({true, {}, {}});
        throw;
      }
    }
    return bq;
  });
}

std::unique_ptr<DataProducer::Iteration> DataBuffer::fetch() {
  NNTR_THROW_IF(!producer, std::runtime_error) << "producer does not exist";
  auto bq = bq_view.lock();
  NNTR_THROW_IF(!bq, std::runtime_error)
    << "Cannot fetch, either fetcher is not running or fetcher has ended and "
       "invalidated";
  auto iteration = bq->wait_and_pop();
  NNTR_THROW_IF(!iteration, std::runtime_error)
    << "fetched iteration is null, should not happen at all cases";

  /// if last equals true, resets bq_view
  if (std::get<0>(*iteration) == true) {
    bq_view.reset();
  }
  return iteration;
}

DataProducer::Generator
DataBuffer::batcher(const std::vector<TensorDim> &input_dims,
                    const std::vector<TensorDim> &label_dims) {
  NNTR_THROW_IF(!producer, std::invalid_argument) << "producer does not exist";
  return producer->finalize(input_dims, label_dims);
}

void DataBuffer::displayProgress(const int count, float loss) {
  int barWidth = 20;
  /** this is temporary measure, will be getting this as an argument */
  int batch_size = 1;
  int samples_per_epoch = 0;

  std::stringstream ssInt;
  ssInt << count * batch_size;

  std::string str = ssInt.str();
  int len = str.size();

  if (samples_per_epoch == 0) {
    int pad_left = (barWidth - len) / 2;
    int pad_right = barWidth - pad_left - len;
    std::string out_str =
      std::string(pad_left, ' ') + str + std::string(pad_right, ' ');
    std::cout << " [ ";
    std::cout << out_str;
    std::cout << " ] "
              << " ( Training Loss: " << loss << " )\r";
  } else {
    float progress;
    if (batch_size > samples_per_epoch)
      progress = 1.0;
    else
      progress = (((float)(count * batch_size)) / (float)samples_per_epoch);

    int pos = barWidth * progress;
    std::cout << " [ ";
    for (int l = 0; l < barWidth; ++l) {
      if (l <= pos)
        std::cout << "=";
      else
        std::cout << " ";
    }
    std::cout << " ] " << int(progress * 100.0) << "% ( Training Loss: " << loss
              << " )\r";
  }

  std::cout.flush();
}

void DataBuffer::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, *db_props);
  if (producer) {
    producer->setProperty(left);
  } else {
    NNTR_THROW_IF(!left.empty(), std::invalid_argument)
      << "[DataBuffer] Failed to set property";
  }
}
} /* namespace nntrainer */
