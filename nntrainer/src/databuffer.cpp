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

#include <assert.h>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <databuffer.h>
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

std::exception_ptr globalExceptionPtr = nullptr;

namespace nntrainer {

std::mutex data_lock;

std::mutex readyTrainData;
std::mutex readyValData;
std::mutex readyTestData;

std::condition_variable cv_train;
std::condition_variable cv_val;
std::condition_variable cv_test;

int DataBuffer::rangeRandom(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

int DataBuffer::run(BufferType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN:
    if (trainReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[DATA_TRAIN]) {
      this->train_running = true;
      this->train_thread = std::thread(&DataBuffer::updateData, this, type);
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    }
    break;
  case BUF_VAL:
    if (valReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;
    if (validation[DATA_VAL]) {
      this->val_running = true;
      this->val_thread = std::thread(&DataBuffer::updateData, this, type);
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    }
    break;
  case BUF_TEST:
    if (testReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[DATA_TEST]) {
      this->test_running = true;
      this->test_thread = std::thread(&DataBuffer::updateData, this, type);
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    }
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return status;
}

int DataBuffer::clear(BufferType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN: {
    train_running = false;
    if (validation[DATA_TRAIN] && true == train_thread.joinable())
      train_thread.join();
    this->train_data.clear();
    this->train_data_label.clear();
    this->cur_train_bufsize = 0;
    this->rest_train = max_train;
    trainReadyFlag = DATA_NOT_READY;
  } break;
  case BUF_VAL: {
    val_running = false;
    if (validation[DATA_VAL] && true == val_thread.joinable())
      val_thread.join();
    this->val_data.clear();
    this->val_data_label.clear();
    this->cur_val_bufsize = 0;
    this->rest_val = max_val;
    valReadyFlag = DATA_NOT_READY;
  } break;
  case BUF_TEST: {
    test_running = false;
    if (validation[DATA_TEST] && true == test_thread.joinable())
      test_thread.join();
    this->test_data.clear();
    this->test_data_label.clear();
    this->cur_test_bufsize = 0;
    this->rest_test = max_test;
    testReadyFlag = DATA_NOT_READY;
  } break;
  default:
    ml_loge("Error: Not Supported Data Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

int DataBuffer::clear() {
  unsigned int i;

  int status = ML_ERROR_NONE;
  for (i = BUF_TRAIN; i <= BUF_TEST; ++i) {
    BufferType type = static_cast<BufferType>(i);
    status = this->clear(type);

    if (status != ML_ERROR_NONE) {
      ml_loge("Error: error occurred during clearing");
      return status;
    }
  }

  return status;
}

bool DataBuffer::getDataFromBuffer(BufferType type, vec_4d &outVec,
                                   vec_4d &outLabel) {
  unsigned int J, i, j, k, L, l;
  unsigned int width = input_dim.width();
  unsigned int height = input_dim.height();
  unsigned int channel = input_dim.channel();

  switch (type) {
  case BUF_TRAIN: {
    std::vector<int> list;
    std::unique_lock<std::mutex> ultrain(readyTrainData);
    cv_train.wait(ultrain, [this]() -> int { return trainReadyFlag; });

    if (train_data.size() < batch_size || trainReadyFlag == DATA_ERROR ||
        trainReadyFlag == DATA_END) {
      return false;
    }

    for (k = 0; k < batch_size; ++k) {
      std::vector<std::vector<std::vector<float>>> v_channel;
      for (l = 0; l < channel; ++l) {
        L = l * width * height;
        std::vector<std::vector<float>> v_height;
        for (j = 0; j < height; ++j) {
          J = L + j * width;
          std::vector<float> v_width;
          for (i = 0; i < width; ++i) {
            v_width.push_back(train_data[k][J + i]);
          }
          v_height.push_back(v_width);
        }
        v_channel.push_back(v_height);
      }
      outVec.push_back(v_channel);
      outLabel.push_back({{train_data_label[k]}});
    }

    data_lock.lock();

    for (i = 0; i < batch_size; ++i) {
      train_data.erase(train_data.begin() + i);
      train_data_label.erase(train_data_label.begin() + i);
      cur_train_bufsize--;
    }
  } break;
  case BUF_VAL: {
    std::vector<int> list;
    std::unique_lock<std::mutex> ulval(readyValData);
    cv_val.wait(ulval, [this]() -> bool { return valReadyFlag; });
    if (val_data.size() < batch_size || valReadyFlag == DATA_ERROR ||
        valReadyFlag == DATA_END) {
      return false;
    }

    for (k = 0; k < batch_size; ++k) {
      std::vector<std::vector<std::vector<float>>> v_channel;
      for (l = 0; l < channel; ++l) {
        L = l * width * height;
        std::vector<std::vector<float>> v_height;
        for (j = 0; j < height; ++j) {
          J = L + j * width;
          std::vector<float> v_width;
          for (i = 0; i < width; ++i) {
            v_width.push_back(val_data[k][J + i]);
          }
          v_height.push_back(v_width);
        }
        v_channel.push_back(v_height);
      }
      outVec.push_back(v_channel);
      outLabel.push_back({{val_data_label[k]}});
    }

    data_lock.lock();

    for (i = 0; i < batch_size; ++i) {
      val_data.erase(val_data.begin() + i);
      val_data_label.erase(val_data_label.begin() + i);
      cur_val_bufsize--;
    }

  } break;
  case BUF_TEST: {
    std::vector<int> list;
    std::unique_lock<std::mutex> ultest(readyTestData);
    cv_test.wait(ultest, [this]() -> bool { return testReadyFlag; });

    if (test_data.size() < batch_size || testReadyFlag == DATA_ERROR ||
        testReadyFlag == DATA_END) {
      return false;
    }

    for (k = 0; k < batch_size; ++k) {
      std::vector<std::vector<std::vector<float>>> v_channel;
      for (l = 0; l < channel; ++l) {
        L = l * width * height;
        std::vector<std::vector<float>> v_height;
        for (j = 0; j < height; ++j) {
          J = L + j * width;
          std::vector<float> v_width;
          for (i = 0; i < width; ++i) {
            v_width.push_back(test_data[k][J + i]);
          }
          v_height.push_back(v_width);
        }
        v_channel.push_back(v_height);
      }
      outVec.push_back(v_channel);
      outLabel.push_back({{test_data_label[k]}});
    }

    data_lock.lock();
    for (i = 0; i < batch_size; ++i) {
      test_data.erase(test_data.begin() + i);
      test_data_label.erase(test_data_label.begin() + i);
      cur_test_bufsize--;
    }
  } break;
  default:
    ml_loge("Error: Not Supported Data Type");
    return false;
    break;
  }

  data_lock.unlock();

  return true;
}

int DataBuffer::setClassNum(unsigned int num) {
  int status = ML_ERROR_NONE;
  if (num <= 0) {
    ml_loge("Error: number of class should be bigger than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (class_num != 0 && class_num != num) {
    ml_loge("Error: number of class should be same with number of label label");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  class_num = num;
  return status;
}

int DataBuffer::setBufSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  /// TODO: move this to initialization than here.
  if (size < batch_size) {
    ml_loge("Error: buffer size must be greater than batch size");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
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

int DataBuffer::setFeatureSize(TensorDim indim) {
  int status = ML_ERROR_NONE;
  input_dim = indim;
  return status;
}

void DataBuffer::displayProgress(const int count, BufferType type, float loss) {
  int barWidth = 20;
  float max_size = max_train;
  switch (type) {
  case BUF_TRAIN:
    max_size = max_train;
    break;
  case BUF_VAL:
    max_size = max_val;
    break;
  case BUF_TEST:
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

} /* namespace nntrainer */
