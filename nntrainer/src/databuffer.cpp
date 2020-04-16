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

#include "databuffer.h"
#include <assert.h>
#include <nntrainer_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "nntrainer_error.h"

namespace nntrainer {

std::mutex data_lock;

std::mutex readyTrainData;
std::mutex readyValData;
std::mutex readyTestData;

std::condition_variable cv_train;
std::condition_variable cv_val;
std::condition_variable cv_test;

bool trainReadyFlag;
bool valReadyFlag;
bool testReadyFlag;

static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

DataBuffer::DataBuffer(int train_num, int val_num, int test_num) {
  this->train_bufsize = train_num;
  this->val_bufsize = val_num;
  this->test_bufsize = test_num;
}

bool DataBuffer::init(int mini_batch, unsigned int train_bufsize, unsigned int val_bufsize, unsigned int test_bufsize,
                      std::ifstream &train_file, std::ifstream &val_file, std::ifstream &test_file,
                      unsigned int max_train, unsigned int max_val, unsigned int max_test, unsigned int in_size,
                      unsigned int c_num) {
  this->input_size = in_size;
  this->class_num = c_num;

  this->cur_train_bufsize = 0;
  this->cur_val_bufsize = 0;
  this->cur_test_bufsize = 0;

  this->train_bufsize = train_bufsize;
  this->val_bufsize = val_bufsize;
  this->test_bufsize = test_bufsize;

  this->mini_batch = mini_batch;

  this->max_train = max_train;
  this->max_val = max_val;
  this->max_test = max_test;

  this->rest_train = max_train;
  this->rest_val = max_val;
  this->rest_test = max_test;

  this->train_running = true;
  this->val_running = true;
  this->test_running = true;

  trainReadyFlag = false;
  valReadyFlag = false;
  testReadyFlag = false;

  return true;
}

void DataBuffer::updateData(BufferType type, std::ifstream &file) {
  unsigned int max_size = 0;
  unsigned int buf_size = 0;
  unsigned int *rest_size = NULL;
  unsigned int *cur_size = NULL;
  bool *running = NULL;
  std::vector<std::vector<float>> *data = NULL;
  std::vector<std::vector<float>> *datalabel = NULL;

  switch (type) {
    case BUF_TRAIN:
      max_size = max_train;
      buf_size = train_bufsize;
      rest_size = &rest_train;
      cur_size = &cur_train_bufsize;
      running = &train_running;
      data = &train_data;
      datalabel = &train_data_label;
      break;
    case BUF_VAL:
      max_size = max_val;
      buf_size = val_bufsize;
      rest_size = &rest_val;
      cur_size = &cur_val_bufsize;
      running = &val_running;
      data = &val_data;
      datalabel = &val_data_label;
      break;
    case BUF_TEST:
      max_size = max_test;
      buf_size = test_bufsize;
      rest_size = &rest_test;
      cur_size = &cur_test_bufsize;
      running = &test_running;
      data = &test_data;
      datalabel = &test_data_label;
      break;
    default:
      break;
  }

  unsigned int I;
  std::vector<unsigned int> mark;
  mark.resize(max_size);
  file.seekg(0, std::ios_base::end);
  uint64_t file_length = file.tellg();

  for (unsigned int i = 0; i < max_size; ++i) {
    mark[i] = i;
  }

  while ((*running) && mark.size() != 0) {
    if (buf_size - (*cur_size) > 0 && (*rest_size) > 0) {
      std::vector<float> vec;
      std::vector<float> veclabel;

      unsigned int id = rangeRandom(0, mark.size() - 1);
      I = mark[id];
      if (I > max_size)
        ml_loge("Error: Test case id cannot exceed maximum number of test");

      mark.erase(mark.begin() + id);
      uint64_t position = (I * input_size + I * class_num) * sizeof(float);

      if (position > file_length || position > ULLONG_MAX)
        ml_loge("Error: Cannot exceed max file size");

      file.seekg(position, std::ios::beg);

      for (unsigned int j = 0; j < input_size; ++j) {
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
      switch (type) {
        case BUF_TRAIN: {
          std::lock_guard<std::mutex> lgtrain(readyTrainData);
          trainReadyFlag = true;
          cv_train.notify_all();
        } break;
        case BUF_VAL: {
          std::lock_guard<std::mutex> lgval(readyValData);
          valReadyFlag = true;
          cv_val.notify_all();
        } break;
        case BUF_TEST: {
          std::lock_guard<std::mutex> lgtest(readyTestData);
          testReadyFlag = true;
          cv_test.notify_all();
        } break;
        default:
          break;
      }
    }
  }
}

void DataBuffer::run(BufferType type, std::ifstream &file) {
  switch (type) {
    case BUF_TRAIN:
      this->train_thread = std::thread(&DataBuffer::updateData, this, BUF_TRAIN, std::ref(file));
      this->train_thread.detach();
      break;
    case BUF_VAL:
      this->val_thread = std::thread(&DataBuffer::updateData, this, BUF_VAL, std::ref(file));
      this->val_thread.detach();
      break;
    case BUF_TEST:
      this->test_thread = std::thread(&DataBuffer::updateData, this, BUF_TEST, std::ref(file));
      this->test_thread.detach();
      break;
    default:
      break;
  }
}

bool DataBuffer::getStatus(BufferType type) {
  int ret = true;
  switch (type) {
    case BUF_TRAIN:
      if ((train_data.size() < mini_batch) && trainReadyFlag)
        ret = false;
      break;
    case BUF_VAL:
      if ((val_data.size() < mini_batch) && valReadyFlag)
        ret = false;
      break;
    case BUF_TEST:
      if ((test_data.size() < mini_batch) && testReadyFlag)
        ret = false;
      break;
    default:
      break;
  }
  return ret;
}

void DataBuffer::clear(BufferType type, std::ifstream &file) {
  switch (type) {
    case BUF_TRAIN: {
      train_running = false;
      this->train_data.clear();
      this->train_data_label.clear();
      this->cur_train_bufsize = 0;
      this->rest_train = max_train;
      trainReadyFlag = false;

      file.clear();
      file.seekg(0, std::ios::beg);

      this->train_running = true;
    } break;
    case BUF_VAL: {
      val_running = false;
      this->val_data.clear();
      this->val_data_label.clear();
      this->cur_val_bufsize = 0;
      this->rest_val = max_val;
      valReadyFlag = false;

      file.clear();
      file.seekg(0, std::ios::beg);

      this->val_running = true;
    } break;
    case BUF_TEST: {
      test_running = false;
      this->test_data.clear();
      this->test_data_label.clear();
      this->cur_test_bufsize = 0;
      this->rest_test = max_test;
      testReadyFlag = false;
      file.clear();
      file.seekg(0, std::ios::beg);
      this->test_running = true;
    } break;
    default:
      break;
  }
}

bool DataBuffer::getDataFromBuffer(BufferType type, std::vector<std::vector<std::vector<float>>> &outVec,
                                   std::vector<std::vector<std::vector<float>>> &outLabel, unsigned int batch,
                                   unsigned int width, unsigned int height, unsigned int c_num) {
  int nomI;
  unsigned int J, i, j, k;

  switch (type) {
    case BUF_TRAIN: {
      std::vector<int> list;

      if (!getStatus(BUF_TRAIN))
        return false;

      {
        std::unique_lock<std::mutex> ultest(readyTrainData);
        cv_train.wait(ultest, []() -> bool { return trainReadyFlag; });
      }

      data_lock.lock();
      for (k = 0; k < batch; ++k) {
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
      for (i = 0; i < batch; ++i) {
        train_data.erase(train_data.begin() + list[i]);
        train_data_label.erase(train_data_label.begin() + list[i]);
        cur_train_bufsize--;
      }
    } break;
    case BUF_VAL: {
      std::vector<int> list;
      if (!getStatus(BUF_VAL))
        return false;

      {
        std::unique_lock<std::mutex> ulval(readyValData);
        cv_val.wait(ulval, []() -> bool { return valReadyFlag; });
      }

      data_lock.lock();
      for (k = 0; k < batch; ++k) {
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
      for (i = 0; i < batch; ++i) {
        val_data.erase(val_data.begin() + list[i]);
        val_data_label.erase(val_data_label.begin() + list[i]);
        cur_val_bufsize--;
      }
    } break;
    case BUF_TEST: {
      std::vector<int> list;
      if (!getStatus(BUF_TEST))
        return false;

      {
        std::unique_lock<std::mutex> ultest(readyTestData);
        cv_test.wait(ultest, []() -> bool { return testReadyFlag; });
      }

      data_lock.lock();
      for (k = 0; k < batch; ++k) {
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
      for (i = 0; i < batch; ++i) {
        test_data.erase(test_data.begin() + list[i]);
        test_data_label.erase(test_data_label.begin() + list[i]);
        cur_test_bufsize--;
      }
    } break;
    default:
      return false;
      break;
  }
  data_lock.unlock();

  return true;
}

int DataBuffer::setDataFile(std::string path, DataType type) {
  int status = ML_ERROR_NONE;
  std::ifstream data_file(path);
  if (!data_file.good()) {
    ml_loge("Error: Cannot open configuraiotn file %s", path.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  switch (type) {
    case DATA_TRAIN:
      train_file = path;
      break;
    case DATA_VAL:
      val_file = path;
      break;
    case DATA_TEST:
      test_file = path;
      break;
    case DATA_LABEL: {
      std::string data;
      while (data_file >> data) {
        labels.push_back(data);
      }

      if (labels.size() != class_num) {
        ml_loge("Error: number of label is not equal to number of class : %d vs. %d", (int)labels.size(), class_num);
        return ML_ERROR_INVALID_PARAMETER;
      }
    } break;
    case DATA_UNKNOWN:
    default:
      ml_loge("Error: Data Type is unknown");
      return ML_ERROR_INVALID_PARAMETER;
      break;
  }
  return status;
}

int DataBuffer::setClassNum(unsigned int num) {
  int status = ML_ERROR_NONE;
  if (num <= 0) {
    ml_loge("Error: number of class should be bigger than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }
  class_num = num;
  return status;
}

} /* namespace nntrainer */
