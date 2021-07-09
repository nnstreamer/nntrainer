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
 * @file	databuffer_file.cpp
 * @date	27 April 2020
 * @brief	This is buffer object take data from raw files
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <assert.h>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <databuffer_file.h>
#include <databuffer_util.h>
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

extern std::exception_ptr globalExceptionPtr;

namespace nntrainer {

extern std::mutex data_lock;

extern std::mutex readyTrainData;
extern std::mutex readyValData;
extern std::mutex readyTestData;

extern std::condition_variable cv_train;
extern std::condition_variable cv_val;
extern std::condition_variable cv_test;

static long getFileSize(std::string file_name) {
  std::ifstream file_stream(file_name.c_str(), std::ios::in | std::ios::binary);
  if (file_stream.good()) {
    file_stream.seekg(0, std::ios::end);
    return file_stream.tellg();
  } else {
    return 0;
  }
}

int DataBufferFromDataFile::init() {
  int status = ML_ERROR_NONE;

  status = DataBuffer::init();
  if (status != ML_ERROR_NONE)
    return status;

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_TRAIN)] &&
      max_train < batch_size) {
    max_train = batch_size;
  }

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_VAL)] &&
      max_val < batch_size) {
    max_val = batch_size;
  }

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_TEST)] &&
      max_test < batch_size) {
    max_test = batch_size;
  }

  this->rest_train = max_train;
  this->rest_val = max_val;
  this->rest_test = max_test;

  this->train_running = true;
  this->val_running = true;
  this->test_running = true;

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_TRAIN)] &&
      max_train < train_bufsize) {
    ml_logw("Warning: Total number of train is less than train buffer size. "
            "Train buffer size is set as total number of train");
    train_bufsize = batch_size;
  }

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_VAL)] &&
      max_val < val_bufsize) {
    ml_logw(
      "Warning: Total number of validation is less than validation buffer "
      "size. Validation buffer size is set as total number of validation");
    val_bufsize = batch_size;
  }

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_TEST)] &&
      max_test < test_bufsize) {
    ml_logw("Warning: Total number of test is less than test buffer size. Test "
            "buffer size is set as total number of test");
    test_bufsize = batch_size;
  }

  return ML_ERROR_NONE;
}

void DataBufferFromDataFile::updateData() {
  auto type = DatasetDataUsageType::DATA_TRAIN;
  unsigned int max_size = 0;
  unsigned int buf_size = 0;
  unsigned int *rest_size = NULL;
  unsigned int *cur_size = NULL;
  bool *running = NULL;
  std::vector<std::vector<float>> *data = NULL;
  std::vector<std::vector<float>> *datalabel = NULL;
  std::ifstream file;
  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN: {
    max_size = max_train;
    buf_size = train_bufsize;
    rest_size = &rest_train;
    cur_size = &cur_train_bufsize;
    running = &train_running;
    data = &train_data;
    datalabel = &train_data_label;

    std::ifstream train_stream(train_name, std::ios::in | std::ios::binary);
    file.swap(train_stream);
    readyTrainData.lock();
    trainReadyFlag = DATA_NOT_READY;
    readyTrainData.unlock();

  } break;
  case DatasetDataUsageType::DATA_VAL: {
    max_size = max_val;
    buf_size = val_bufsize;
    rest_size = &rest_val;
    cur_size = &cur_val_bufsize;
    running = &val_running;
    data = &val_data;
    datalabel = &val_data_label;

    std::ifstream val_stream(val_name, std::ios::in | std::ios::binary);
    file.swap(val_stream);
    readyValData.lock();
    valReadyFlag = DATA_NOT_READY;
    readyValData.unlock();

  } break;
  case DatasetDataUsageType::DATA_TEST: {
    max_size = max_test;
    buf_size = test_bufsize;
    rest_size = &rest_test;
    cur_size = &cur_test_bufsize;
    running = &test_running;
    data = &test_data;
    datalabel = &test_data_label;

    std::ifstream test_stream(test_name, std::ios::in | std::ios::binary);
    file.swap(test_stream);
    readyTestData.lock();
    testReadyFlag = DATA_NOT_READY;
    readyTestData.unlock();
  } break;
  default:
    try {
      throw std::runtime_error("Error: Not Supported Data Type");
    } catch (...) {
      globalExceptionPtr = std::current_exception();
      NN_EXCEPTION_NOTI(DATA_ERROR);
      return;
    }
    break;
  }

  unsigned int I;
  std::vector<unsigned int> mark;
  mark.resize(max_size);
  file.clear();
  file.seekg(0, std::ios_base::end);
  uint64_t file_length = file.tellg();

  for (unsigned int i = 0; i < max_size; ++i) {
    mark[i] = i;
  }

  while ((*running)) {

    if (mark.size() == 0) {
      NN_EXCEPTION_NOTI(DATA_END);
      break;
    }

    if (buf_size - (*cur_size) > 0 && (*rest_size) > 0) {
      std::vector<float> vec;
      std::vector<float> veclabel;

      unsigned int id = rangeRandom(0, mark.size() - 1);
      I = mark[id];

      try {
        if (I > max_size) {
          throw std::runtime_error(
            "Error: Test case id cannot exceed maximum number of test");
        }
      } catch (...) {
        globalExceptionPtr = std::current_exception();
        NN_EXCEPTION_NOTI(DATA_ERROR);
        return;
      }

      mark.erase(mark.begin() + id);

      uint64_t position =
        (uint64_t)((I * input_dim.getFeatureLen() + (uint64_t)I * class_num) *
                   sizeof(float));
      try {
        if (position > file_length) {
          throw std::runtime_error("Error: Cannot exceed max file size");
        }
      } catch (...) {
        globalExceptionPtr = std::current_exception();
        NN_EXCEPTION_NOTI(DATA_ERROR);
        return;
      }

      file.seekg(position, std::ios::beg);

      for (unsigned int j = 0; j < input_dim.getFeatureLen(); ++j) {
        float d;
        file.read((char *)&d, sizeof(float));
        vec.push_back(d);
      }

      for (unsigned int j = 0; j < class_num; ++j) {
        float d;
        file.read((char *)&d, sizeof(float));
        veclabel.push_back(d);
      }

      data_lock.lock();
      data->push_back(vec);
      datalabel->push_back(veclabel);
      (*rest_size)--;
      (*cur_size)++;
      data_lock.unlock();
    }
    if (buf_size == (*cur_size)) {
      NN_EXCEPTION_NOTI(DATA_READY);
    }
  }

  file.close();
}

int DataBufferFromDataFile::setDataFile(const std::string &path) {
  auto type = DatasetDataUsageType::DATA_TRAIN;
  int status = ML_ERROR_NONE;
  std::ifstream data_file(path.c_str());

  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN: {
    validation[static_cast<int>(type)] = true;
    if (!data_file.good()) {
      ml_loge(
        "Error: Cannot open data file, Datafile is necessary for training");
      validation[static_cast<int>(type)] = false;
      return ML_ERROR_INVALID_PARAMETER;
    }
    train_name = path;
  } break;
  case DatasetDataUsageType::DATA_VAL: {
    validation[static_cast<int>(type)] = true;
    if (!data_file.good()) {
      ml_loge("Error: Cannot open validation data file. Cannot validate "
              "training result");
      validation[static_cast<int>(type)] = false;
      return ML_ERROR_INVALID_PARAMETER;
    }
    val_name = path;
  } break;
  case DatasetDataUsageType::DATA_TEST: {
    validation[static_cast<int>(type)] = true;
    if (!data_file.good()) {
      ml_loge("Error: Cannot open test data file. Cannot test training result");
      validation[static_cast<int>(type)] = false;
      return ML_ERROR_INVALID_PARAMETER;
    }
    test_name = path;
  } break;
  case DatasetDataUsageType::DATA_UNKNOWN:
  default:
    ml_loge("Error: Not Supported Data Type");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
    break;
  }
  ml_logd("datafile has set. type: %d, path: %s", static_cast<int>(type),
          path.c_str());

  return status;
}

int DataBufferFromDataFile::setFeatureSize(TensorDim tdim) {
  int status = ML_ERROR_NONE;
  long file_size = 0;

  status = DataBuffer::setFeatureSize(tdim);
  if (status != ML_ERROR_NONE)
    return status;

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_TRAIN)]) {
    file_size = getFileSize(train_name);
    max_train = static_cast<unsigned int>(
      file_size /
      (class_num * sizeof(int) + input_dim.getFeatureLen() * sizeof(float)));
    if (max_train < batch_size) {
      ml_logw("Warning: number of training data is smaller than batch size");
    }
  } else {
    max_train = 0;
  }

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_VAL)]) {
    file_size = getFileSize(val_name);
    max_val = static_cast<unsigned int>(
      file_size /
      (class_num * sizeof(int) + input_dim.getFeatureLen() * sizeof(float)));
    if (max_val < batch_size) {
      ml_logw("Warning: number of val data is smaller than batch size");
    }
  } else {
    max_val = 0;
  }

  if (validation[static_cast<int>(DatasetDataUsageType::DATA_TEST)]) {
    file_size = getFileSize(test_name);
    max_test = static_cast<unsigned int>(
      file_size /
      (class_num * sizeof(int) + input_dim.getFeatureLen() * sizeof(float)));
    if (max_test < batch_size) {
      ml_logw("Warning: number of test data is smaller than batch size");
    }
  } else {
    max_test = 0;
  }

  return status;
}

} /* namespace nntrainer */
