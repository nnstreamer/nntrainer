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
  PropsBufferSize(unsigned int value = 1) :
    nntrainer::Property<unsigned int>(value) {}
  bool isValid(const unsigned int &v) { return v > 0; }
  static constexpr const char *key = "buffer_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

constexpr char USER_DATA[] = "user_data";

DataBuffer::DataBuffer(DatasetType type) :
  data_buffer_type(type),
  batch_producer(),
  data_q(),
  label_q(),
  queue_status(DataStatus::DATA_NOT_READY),
  buf_size(0),
  cur_bufsize(0),
  class_num(0),
  batch_size(0),
  samples_per_epoch(0),
  remaining_samples_per_epoch(0),
  is_running(false),
  initialized(false),
  producer(nullptr),
  db_props(new Props()) {
  rng.seed(getSeed());
};

DataBuffer::DataBuffer(std::unique_ptr<DataProducer> &&producer_) :
  producer(std::move(producer_)),
  db_props(new Props()) {
  rng.seed(getSeed());
}

DataBuffer::~DataBuffer(){};

int DataBuffer::rangeRandom(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

int DataBuffer::run() {
  if (queue_status == DataStatus::DATA_ERROR)
    return ML_ERROR_INVALID_PARAMETER;

  if (initialized) {
    this->is_running = true;
    this->batch_producer = std::thread(&DataBuffer::updateData, this);
    if (consumer_exception_ptr) {
      try {
        std::rethrow_exception(consumer_exception_ptr);
      } catch (const std::exception &ex) {
        std::cout << ex.what() << "\n";
        return ML_ERROR_INVALID_PARAMETER;
      }
    }
  } else {
    ml_loge("Error: Training Data Set is not valid");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int DataBuffer::clear() {
  int status = ML_ERROR_NONE;
  setStateAndNotify(DataStatus::DATA_NOT_READY);
  is_running = false;
  if (initialized && true == batch_producer.joinable())
    batch_producer.join();
  this->data_q.clear();
  this->label_q.clear();
  this->cur_bufsize = 0;
  this->remaining_samples_per_epoch = samples_per_epoch;
  return status;
}

bool DataBuffer::getDataFromBuffer(float *out, float *label) {
  using QueueType = std::vector<std::vector<float>>;

  auto wait_for_data_fill = [](std::mutex &ready_mutex,
                               std::condition_variable &cv, DataStatus &flag,
                               const unsigned int batch_size,
                               QueueType &queue) {
    while (true) {
      std::unique_lock<std::mutex> ul(ready_mutex);
      cv.wait(ul, [&]() -> bool { return flag != DataStatus::DATA_NOT_READY; });
      if (flag == DataStatus::DATA_ERROR || flag == DataStatus::DATA_END)
        return queue.size() < batch_size ? false : true;

      if (flag == DataStatus::DATA_READY && queue.size() >= batch_size)
        return true;
    }

    throw std::logic_error("[getDataFromBuffer] control should not reach here");
  };

  auto fill_bundled_data_from_queue =
    [](std::mutex &q_lock, QueueType &q, const unsigned int batch_size,
       const unsigned int feature_size, float *buf) {
      for (unsigned int b = 0; b < batch_size; ++b)
        std::copy(q[b].begin(), q[b].begin() + feature_size,
                  buf + b * feature_size);

      q_lock.lock();
      q.erase(q.begin(), q.begin() + batch_size);
      q_lock.unlock();
    };

  /// facade that wait for the databuffer to be filled and pass it to outparam
  /// note that batch_size is passed as an argument because it can vary by
  /// DatasetDataUsageType::BUF_TYPE later...
  auto fill_out_params =
    [&](std::mutex &ready_mutex, std::condition_variable &cv, DataStatus &flag,
        QueueType &data_q, QueueType &label_q, const unsigned int batch_size,
        unsigned int &cur_bufsize) {
      if (!wait_for_data_fill(ready_mutex, cv, flag, batch_size, data_q)) {
        return false;
      }

      fill_bundled_data_from_queue(data_lock, data_q, batch_size,
                                   this->input_dim.getFeatureLen(), out);
      fill_bundled_data_from_queue(data_lock, label_q, batch_size,
                                   this->class_num, label);

      cur_bufsize -= batch_size;
      return true;
    };

  return fill_out_params(status_mutex, cv_status, queue_status, data_q, label_q,
                         batch_size, cur_bufsize);
}

int DataBuffer::setClassNum(unsigned int num) {
  int status = ML_ERROR_NONE;
  if (num == 0) {
    ml_loge("Error: number of class should be bigger than 0");
    initialized = false;
    return ML_ERROR_INVALID_PARAMETER;
  }
  class_num = num;
  return status;
}

int DataBuffer::setBatchSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    initialized = false;
    return ML_ERROR_INVALID_PARAMETER;
  }
  batch_size = size;
  return status;
}

int DataBuffer::init() {
  if (batch_size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    initialized = false;
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (buf_size < batch_size) {
    if (buf_size > 1) {
      ml_logw("Dataset buffer size reset to be at least batch size");
    }
    buf_size = batch_size;
  }

  if (!class_num) {
    ml_loge("Error: number of class must be set");
    initialized = false;
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!this->input_dim.getFeatureLen()) {
    ml_loge("Error: feature size must be set");
    initialized = false;
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->cur_bufsize = 0;

  setStateAndNotify(DataStatus::DATA_NOT_READY);
  return ML_ERROR_NONE;
}

std::future<std::shared_ptr<BatchQueue>>
DataBuffer::startFetchWorker(const std::vector<TensorDim> &input_dims,
                             const std::vector<TensorDim> &label_dims) {
  NNTR_THROW_IF(!producer, std::invalid_argument) << "producer does not exist";
  auto bq = std::make_shared<BatchQueue>(std::get<PropsBufferSize>(*db_props));
  auto generator = producer->finalize(input_dims, label_dims);
  bq_view = bq;

  return std::async(std::launch::async, [bq, generator] {
    while (true) {
      auto iteration = generator();
      auto last = std::get<0>(iteration);
      bq->wait_and_push(std::move(iteration));
      if (last == true) {
        break;
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

int DataBuffer::setFeatureSize(const TensorDim &indim) {
  int status = ML_ERROR_NONE;
  input_dim = indim;
  return status;
}

void DataBuffer::displayProgress(const int count, float loss) {
  int barWidth = 20;

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

int DataBuffer::setProperty(std::vector<void *> values) {
  int status = ML_ERROR_NONE;
  std::vector<std::string> properties;

  for (unsigned int i = 0; i < values.size(); ++i) {
    char *key_ptr = (char *)values[i];
    std::string key = key_ptr;
    std::string value;

    /** Handle the user_data as a special case */
    if (key == USER_DATA) {
      /** This ensures that a valid user_data element is passed by the user */
      if (i + 1 >= values.size())
        return ML_ERROR_INVALID_PARAMETER;

      this->user_data = values[i + 1];

      /** As values of i+1 is consumed, increase i by 1 */
      i++;
    } else {
      properties.push_back(key);
      continue;
    }
  }

  setProperty(properties);

  return status;
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

int DataBuffer::setGeneratorFunc(datagen_cb func, void *user_data) {
  return ML_ERROR_NOT_SUPPORTED;
}

int DataBuffer::setDataFile(const std::string &path) {
  return ML_ERROR_NOT_SUPPORTED;
}

void DataBuffer::setStateAndNotify(const DataStatus status) {
  std::lock_guard<std::mutex> lgtrain(status_mutex);
  queue_status = status;
  cv_status.notify_all();
}

} /* namespace nntrainer */
