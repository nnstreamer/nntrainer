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

#include <cassert>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <databuffer.h>
#include <databuffer_util.h>
#include <functional>
#include <iomanip>
#include <mutex>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <util_func.h>

std::exception_ptr globalExceptionPtr = nullptr;

namespace nntrainer {

constexpr char USER_DATA[] = "user_data";

std::mutex data_lock;

std::mutex readyTrainData;
std::mutex readyValData;
std::mutex readyTestData;

std::condition_variable cv_train;
std::condition_variable cv_val;
std::condition_variable cv_test;

DataBuffer::DataBuffer(DatasetType type) :
  train_running(),
  val_running(),
  test_running(),
  train_thread(),
  val_thread(),
  test_thread(),
  data_buffer_type(type),
  user_data(nullptr) {
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
  batch_size = 0;
  train_running = false;
  val_running = false;
  test_running = false;
  trainReadyFlag = DATA_NOT_READY;
  valReadyFlag = DATA_NOT_READY;
  testReadyFlag = DATA_NOT_READY;
  rng.seed(getSeed());
};

int DataBuffer::rangeRandom(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

int DataBuffer::run() {
  auto type = DatasetDataUsageType::DATA_TRAIN;
  int status = ML_ERROR_NONE;
  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN:
    if (trainReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[static_cast<int>(DatasetDataUsageType::DATA_TRAIN)]) {
      this->train_running = true;
      this->train_thread = std::thread(&DataBuffer::updateData, this);
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    } else {
      ml_loge("Error: Training Data Set is not valid");
      return ML_ERROR_INVALID_PARAMETER;
    }
    break;
  case DatasetDataUsageType::DATA_VAL:
    if (valReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;
    if (validation[static_cast<int>(DatasetDataUsageType::DATA_VAL)]) {
      this->val_running = true;
      this->val_thread = std::thread(&DataBuffer::updateData, this);
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    } else {
      ml_loge("Error: Validation Data Set is not valid");
      return ML_ERROR_INVALID_PARAMETER;
    }
    break;
  case DatasetDataUsageType::DATA_TEST:
    if (testReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[static_cast<int>(DatasetDataUsageType::DATA_TEST)]) {
      this->test_running = true;
      this->test_thread = std::thread(&DataBuffer::updateData, this);
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    } else {
      ml_loge("Error: Test Data Set is not valid");
      return ML_ERROR_INVALID_PARAMETER;
    }
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return status;
}

int DataBuffer::clear() {
  auto type = DatasetDataUsageType::DATA_TRAIN;
  int status = ML_ERROR_NONE;
  NN_EXCEPTION_NOTI(DATA_NOT_READY);
  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN: {
    train_running = false;
    if (validation[static_cast<int>(DatasetDataUsageType::DATA_TRAIN)] &&
        true == train_thread.joinable())
      train_thread.join();
    this->train_data.clear();
    this->train_data_label.clear();
    this->cur_train_bufsize = 0;
    this->rest_train = max_train;
  } break;
  case DatasetDataUsageType::DATA_VAL: {
    val_running = false;
    if (validation[static_cast<int>(DatasetDataUsageType::DATA_VAL)] &&
        true == val_thread.joinable())
      val_thread.join();
    this->val_data.clear();
    this->val_data_label.clear();
    this->cur_val_bufsize = 0;
    this->rest_val = max_val;
  } break;
  case DatasetDataUsageType::DATA_TEST: {
    test_running = false;
    if (validation[static_cast<int>(DatasetDataUsageType::DATA_TEST)] &&
        true == test_thread.joinable())
      test_thread.join();
    this->test_data.clear();
    this->test_data_label.clear();
    this->cur_test_bufsize = 0;
    this->rest_test = max_test;
  } break;
  default:
    ml_loge("Error: Not Supported Data Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

bool DataBuffer::getDataFromBuffer(float *out, float *label) {
  auto type = DatasetDataUsageType::DATA_TRAIN;
  using QueueType = std::vector<std::vector<float>>;

  auto wait_for_data_fill = [](std::mutex &ready_mutex,
                               std::condition_variable &cv, DataStatus &flag,
                               const unsigned int batch_size,
                               QueueType &queue) {
    while (true) {
      std::unique_lock<std::mutex> ul(ready_mutex);
      cv.wait(ul, [&]() -> bool { return flag; });
      if (flag == DATA_ERROR || flag == DATA_END)
        return queue.size() < batch_size ? false : true;

      if (flag == DATA_READY && queue.size() >= batch_size)
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

  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN:
    if (!fill_out_params(readyTrainData, cv_train, trainReadyFlag, train_data,
                         train_data_label, batch_size, cur_train_bufsize))
      return false;
    break;
  case DatasetDataUsageType::DATA_VAL:
    if (!fill_out_params(readyValData, cv_val, valReadyFlag, val_data,
                         val_data_label, batch_size, cur_val_bufsize))
      return false;
    break;
  case DatasetDataUsageType::DATA_TEST:
    if (!fill_out_params(readyTestData, cv_test, testReadyFlag, test_data,
                         test_data_label, batch_size, cur_test_bufsize))
      return false;
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    return false;
    break;
  }

  return true;
}

int DataBuffer::setClassNum(unsigned int num) {
  int status = ML_ERROR_NONE;
  if (num == 0) {
    ml_loge("Error: number of class should be bigger than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  class_num = num;
  return status;
}

int DataBuffer::setBufSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  train_bufsize = size;
  val_bufsize = size;
  test_bufsize = size;
  return status;
}

int DataBuffer::setBatchSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  batch_size = size;
  return status;
}

int DataBuffer::init() {
  if (batch_size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** for now, train_bufsize, val_bufsize and test_bufsize are same value */
  if (train_bufsize < batch_size) {
    if (train_bufsize > 1) {
      ml_logw("Dataset buffer size reset to be at least batch size");
    }
    train_bufsize = batch_size;
    val_bufsize = batch_size;
    test_bufsize = batch_size;
  }

  if (!class_num) {
    ml_loge("Error: number of class must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!this->input_dim.getFeatureLen()) {
    ml_loge("Error: feature size must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->cur_train_bufsize = 0;
  this->cur_val_bufsize = 0;
  this->cur_test_bufsize = 0;

  readyTrainData.lock();
  trainReadyFlag = DATA_NOT_READY;
  readyTrainData.unlock();
  readyValData.lock();
  valReadyFlag = DATA_NOT_READY;
  readyValData.unlock();
  readyTestData.lock();
  testReadyFlag = DATA_NOT_READY;
  readyTestData.unlock();
  return ML_ERROR_NONE;
}

int DataBuffer::setFeatureSize(TensorDim indim) {
  int status = ML_ERROR_NONE;
  input_dim = indim;
  return status;
}

void DataBuffer::displayProgress(const int count, float loss) {
  auto type = DatasetDataUsageType::DATA_TRAIN;
  int barWidth = 20;
  float max_size = max_train;
  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN:
    max_size = max_train;
    break;
  case DatasetDataUsageType::DATA_VAL:
    max_size = max_val;
    break;
  case DatasetDataUsageType::DATA_TEST:
    max_size = max_test;
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    break;
  }
  std::stringstream ssInt;
  ssInt << count * batch_size;

  std::string str = ssInt.str();
  int len = str.length();

  if (max_size == 0) {
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
    if (batch_size > max_size)
      progress = 1.0;
    else
      progress = (((float)(count * batch_size)) / max_size);

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

  status = setProperty(properties);

  return status;
}

int DataBuffer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseDataProperty(key);
    if (value.empty())
      return ML_ERROR_INVALID_PARAMETER;

    status = setProperty(static_cast<PropertyType>(type), value);
    NN_RETURN_STATUS();
  }

  return status;
}

int DataBuffer::setProperty(const PropertyType type, std::string &value) {
  int status = ML_ERROR_NONE;
  unsigned int size = 0;

  switch (type) {
  case PropertyType::buffer_size:
    status = setUint(size, value);
    NN_RETURN_STATUS();
    status = this->setBufSize(size);
    NN_RETURN_STATUS();
    break;
  default:
    ml_loge("Error: Unknown Data Buffer Property Key");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return status;
}

int DataBuffer::setGeneratorFunc(datagen_cb func, void *user_data) {
  return ML_ERROR_NOT_SUPPORTED;
}

int DataBuffer::setDataFile(const std::string &path) {
  return ML_ERROR_NOT_SUPPORTED;
}

} /* namespace nntrainer */
