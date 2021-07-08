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
#include <databuffer_func.h>
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

int DataBufferFromCallback::init() {
  int status = ML_ERROR_NONE;

  status = DataBuffer::init();
  if (status != ML_ERROR_NONE)
    return status;

  if (callback_train == nullptr)
    return ML_ERROR_BAD_ADDRESS;

  this->max_train = 0;
  this->max_val = 0;
  this->max_test = 0;

  if (train_bufsize > batch_size || train_bufsize == 0) {
    train_bufsize = batch_size;
  }

  if (val_bufsize > batch_size || val_bufsize == 0) {
    val_bufsize = batch_size;
  }

  if (test_bufsize > batch_size || test_bufsize == 0) {
    test_bufsize = batch_size;
  }

  this->train_running = true;
  this->val_running = true;
  this->test_running = true;

  return ML_ERROR_NONE;
}

int DataBufferFromCallback::setGeneratorFunc(DatasetDataUsageType type,
                                             datagen_cb func) {

  int status = ML_ERROR_NONE;
  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN:
    if (!func)
      return ML_ERROR_INVALID_PARAMETER;
    callback_train = func;
    if (func)
      validation[0] = true;
    break;
  case DatasetDataUsageType::DATA_VAL:
    callback_val = func;
    if (func)
      validation[1] = true;
    break;
  case DatasetDataUsageType::DATA_TEST:
    callback_test = func;
    if (func)
      validation[2] = true;
    break;
  default:
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return status;
}

void DataBufferFromCallback::updateData(DatasetDataUsageType type) {
  int status = ML_ERROR_NONE;

  unsigned int buf_size = 0;
  unsigned int *cur_size = NULL;
  bool *running = NULL;
  std::vector<std::vector<float>> *data = NULL;
  std::vector<std::vector<float>> *datalabel = NULL;
  datagen_cb callback;

  switch (type) {
  case DatasetDataUsageType::DATA_TRAIN: {
    buf_size = train_bufsize;
    cur_size = &cur_train_bufsize;
    running = &train_running;
    data = &train_data;
    datalabel = &train_data_label;
    callback = callback_train;
  } break;
  case DatasetDataUsageType::DATA_VAL: {
    buf_size = val_bufsize;
    cur_size = &cur_val_bufsize;
    running = &val_running;
    data = &val_data;
    datalabel = &val_data_label;
    callback = callback_val;
  } break;
  case DatasetDataUsageType::DATA_TEST: {
    buf_size = test_bufsize;
    cur_size = &cur_test_bufsize;
    running = &test_running;
    data = &test_data;
    datalabel = &test_data_label;
    callback = callback_test;
  } break;
  default:
    break;
  }

  try {
    if ((cur_size == NULL) || (running == NULL) || (data == NULL) ||
        (datalabel == NULL))
      throw std::runtime_error("Error: assigning error");
  } catch (...) {
    globalExceptionPtr = std::current_exception();
    NN_EXCEPTION_NOTI(DATA_ERROR);
    return;
  }
  bool endflag = false;

  float **vec_arr = (float **)malloc(sizeof(float *) * 1);
  float **veclabel_arr = (float **)malloc(sizeof(float *) * 1);

  float *vec =
    (float *)malloc(sizeof(float) * input_dim.batch() * input_dim.channel() *
                    input_dim.height() * input_dim.width());
  float *veclabel =
    (float *)malloc(sizeof(float) * input_dim.batch() * class_num);

  try {
    if (vec_arr == nullptr || veclabel_arr == nullptr || vec == nullptr ||
        veclabel == nullptr) {
      free(vec);
      free(veclabel);
      free(vec_arr);
      free(veclabel_arr);
      throw std::runtime_error("Error: assigning error");
    }
  } catch (...) {
    globalExceptionPtr = std::current_exception();
    NN_EXCEPTION_NOTI(DATA_ERROR);
    return;
  }

  vec_arr[0] = vec;
  veclabel_arr[0] = veclabel;

  while ((*running)) {
    endflag = false;
    NN_EXCEPTION_NOTI(DATA_NOT_READY);
    if (buf_size - (*cur_size) > 0) {
      /** @todo Update to support multiple inputs later */
      status = callback(vec_arr, veclabel_arr, &endflag, user_data);
      if (endflag) {
        NN_EXCEPTION_NOTI(DATA_END);
        free(vec);
        free(veclabel);
        free(vec_arr);
        free(veclabel_arr);
        return;
      }
      if (status != ML_ERROR_NONE) {
        NN_EXCEPTION_NOTI(DATA_ERROR);
        free(vec);
        free(veclabel);
        free(vec_arr);
        free(veclabel_arr);
        return;
      }

      for (unsigned int i = 0; i < input_dim.batch(); ++i) {
        std::vector<float> v;
        std::vector<float> vl;
        unsigned int I =
          i * input_dim.channel() * input_dim.height() * input_dim.width();
        for (unsigned int j = 0; j < input_dim.channel(); ++j) {
          unsigned int J = j * input_dim.height() * input_dim.width();
          for (unsigned int k = 0; k < input_dim.height() * input_dim.width();
               ++k) {
            unsigned int K = I + J + k;
            v.push_back(vec[K]);
          }
        }

        I = i * class_num;
        for (unsigned int j = 0; j < class_num; ++j) {
          vl.push_back(veclabel[I + j]);
        }

        data_lock.lock();
        data->push_back(v);
        datalabel->push_back(vl);
        (*cur_size)++;
        data_lock.unlock();
      }
    }
    if (buf_size == (*cur_size)) {
      NN_EXCEPTION_NOTI(DATA_READY);
    }
  }

  free(vec);
  free(veclabel);
  free(vec_arr);
  free(veclabel_arr);
}

int DataBufferFromCallback::setProperty(const PropertyType type,
                                        std::string &value) {
  return DataBuffer::setProperty(type, value);
}

} /* namespace nntrainer */
