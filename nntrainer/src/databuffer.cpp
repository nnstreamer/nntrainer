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

DataStatus trainReadyFlag;
DataStatus valReadyFlag;
DataStatus testReadyFlag;

int DataBuffer::rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

int DataBuffer::run(BufferType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN:
    if (trainReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[DATA_TRAIN]) {
      this->train_running = true;
      this->train_thread =
        std::thread(&DataBuffer::updateData, this, type, std::ref(status));
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
      this->val_thread =
        std::thread(&DataBuffer::updateData, this, type, std::ref(status));
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
      this->test_thread =
        std::thread(&DataBuffer::updateData, this, type, std::ref(status));
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

DataStatus DataBuffer::getStatus(BufferType type) {
  DataStatus ret = DATA_READY;
  if (globalExceptionPtr)
    ret = DATA_ERROR;

  switch (type) {
  case BUF_TRAIN:
    if ((train_data.size() < mini_batch) && trainReadyFlag)
      ret = DATA_NOT_READY;
    break;
  case BUF_VAL:
    if ((val_data.size() < mini_batch) && valReadyFlag)
      ret = DATA_NOT_READY;
    break;
  case BUF_TEST:
    if ((test_data.size() < mini_batch) && testReadyFlag)
      ret = DATA_NOT_READY;
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    ret = DATA_ERROR;
    break;
  }
  return ret;
}

bool DataBuffer::getDataFromBuffer(BufferType type, vec_3d &outVec,
                                   vec_3d &outLabel) {
  int nomI;
  unsigned int J, i, j, k;
  unsigned int width = input_size;
  unsigned int height = 1;

  switch (type) {
  case BUF_TRAIN: {
    std::vector<int> list;

    if (getStatus(BUF_TRAIN) != DATA_READY)
      return false;

    {
      std::unique_lock<std::mutex> ultest(readyTrainData);
      cv_train.wait(ultest, []() -> int { return trainReadyFlag; });
    }

    if (trainReadyFlag == DATA_ERROR) {
      return false;
    }

    data_lock.lock();
    for (k = 0; k < mini_batch; ++k) {
      nomI = rangeRandom(0, train_data.size() - 1);
      std::vector<std::vector<float>> v_height;
      for (j = 0; j < height; ++j) {
        J = j * width;
        std::vector<float> v_width;
        for (i = 0; i < width; ++i) {
          v_width.push_back(train_data[nomI][J + i]);
        }
        v_height.push_back(v_width);
      }

      list.push_back(nomI);
      outVec.push_back(v_height);
      outLabel.push_back({train_data_label[nomI]});
    }
    for (i = 0; i < mini_batch; ++i) {
      train_data.erase(train_data.begin() + list[i]);
      train_data_label.erase(train_data_label.begin() + list[i]);
      cur_train_bufsize--;
    }
  } break;
  case BUF_VAL: {
    std::vector<int> list;
    if (getStatus(BUF_VAL) != DATA_READY)
      return false;

    {
      std::unique_lock<std::mutex> ulval(readyValData);
      cv_val.wait(ulval, []() -> bool { return valReadyFlag; });
    }

    data_lock.lock();
    for (k = 0; k < mini_batch; ++k) {
      nomI = rangeRandom(0, val_data.size() - 1);
      std::vector<std::vector<float>> v_height;
      for (j = 0; j < height; ++j) {
        J = j * width;
        std::vector<float> v_width;
        for (i = 0; i < width; ++i) {
          v_width.push_back(val_data[nomI][J + i]);
        }
        v_height.push_back(v_width);
      }

      list.push_back(nomI);
      outVec.push_back(v_height);
      outLabel.push_back({val_data_label[nomI]});
    }
    for (i = 0; i < mini_batch; ++i) {
      val_data.erase(val_data.begin() + list[i]);
      val_data_label.erase(val_data_label.begin() + list[i]);
      cur_val_bufsize--;
    }
  } break;
  case BUF_TEST: {
    std::vector<int> list;
    if (getStatus(BUF_TEST) != DATA_READY)
      return false;

    {
      std::unique_lock<std::mutex> ultest(readyTestData);
      cv_test.wait(ultest, []() -> bool { return testReadyFlag; });
    }

    data_lock.lock();
    for (k = 0; k < mini_batch; ++k) {
      nomI = rangeRandom(0, test_data.size() - 1);
      std::vector<std::vector<float>> v_height;
      for (j = 0; j < height; ++j) {
        J = j * width;
        std::vector<float> v_width;
        for (i = 0; i < width; ++i) {
          v_width.push_back(test_data[nomI][J + i]);
        }
        v_height.push_back(v_width);
      }

      list.push_back(nomI);
      outVec.push_back(v_height);
      outLabel.push_back({test_data_label[nomI]});
    }
    for (i = 0; i < mini_batch; ++i) {
      test_data.erase(test_data.begin() + list[i]);
      test_data_label.erase(test_data_label.begin() + list[i]);
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
  if (size < mini_batch) {
    ml_loge("Error: buffer size must be greater than batch size");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  train_bufsize = size;
  val_bufsize = size;
  test_bufsize = size;
  return status;
}

int DataBuffer::setMiniBatch(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  mini_batch = size;
  return status;
}

int DataBuffer::setFeatureSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  input_size = size;
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
  ssInt << count * mini_batch;

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
    if (mini_batch > max_size)
      progress = 1.0;
    else
      progress = (((float)(count * mini_batch)) / max_size);

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

} /* namespace nntrainer */
