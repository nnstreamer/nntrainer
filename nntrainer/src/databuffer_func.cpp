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

extern DataStatus trainReadyFlag;
extern DataStatus valReadyFlag;
extern DataStatus testReadyFlag;

int DataBufferFromCallback::init() {
  int status = ML_ERROR_NONE;

  if (!class_num) {
    ml_loge("Error: number of class must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!this->input_dim.getFeatureLen()) {
    ml_loge("Error: featuer size must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->cur_train_bufsize = 0;
  this->cur_val_bufsize = 0;
  this->cur_test_bufsize = 0;

  this->max_train = 0;
  this->max_val = 0;
  this->max_test = 0;

  if (mini_batch == 0) {
    ml_loge("Error: mini batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (train_bufsize > mini_batch) {
    train_bufsize = mini_batch;
  }

  if (val_bufsize > mini_batch) {
    val_bufsize = mini_batch;
  }

  if (test_bufsize > mini_batch) {
    test_bufsize = mini_batch;
  }

  this->train_running = true;
  this->val_running = true;
  this->test_running = true;

  trainReadyFlag = DATA_NOT_READY;
  valReadyFlag = DATA_NOT_READY;
  testReadyFlag = DATA_NOT_READY;

  return status;
}

int DataBufferFromCallback::setFunc(
  BufferType type, std::function<bool(float *, float *, int *)> func) {

  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN:
    callback_train = func;
    if (func)
      validation[0] = true;
    break;
  case BUF_VAL:
    callback_val = func;
    if (func)
      validation[1] = true;
    break;
  case BUF_TEST:
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

void DataBufferFromCallback::updateData(BufferType type, int &status) {
  status = ML_ERROR_NONE;

  unsigned int buf_size = 0;
  unsigned int *cur_size = NULL;
  bool *running = NULL;
  std::vector<std::vector<float>> *data = NULL;
  std::vector<std::vector<float>> *datalabel = NULL;
  std::function<bool(float *, float *, int *)> callback;

  switch (type) {
  case BUF_TRAIN: {
    buf_size = train_bufsize;
    cur_size = &cur_train_bufsize;
    running = &train_running;
    data = &train_data;
    datalabel = &train_data_label;
    callback = callback_train;
  } break;
  case BUF_VAL: {
    buf_size = val_bufsize;
    cur_size = &cur_val_bufsize;
    running = &val_running;
    data = &val_data;
    datalabel = &val_data_label;
    callback = callback_val;
  } break;
  case BUF_TEST: {
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
      throw std::runtime_error("Error: assining error");
  } catch (...) {
    globalExceptionPtr = std::current_exception();
    NN_EXCEPTION_NOTI(DATA_ERROR);
    return;
  }
  bool endflag = false;

  float *vec =
    (float *)malloc(sizeof(float) * input_dim.batch() * input_dim.channel() *
                    input_dim.height() * input_dim.width());
  float *veclabel =
    (float *)malloc(sizeof(float) * input_dim.batch() * class_num);

  while ((*running)) {
    trainReadyFlag = DATA_NOT_READY;
    valReadyFlag = DATA_NOT_READY;
    testReadyFlag = DATA_NOT_READY;
    if (buf_size - (*cur_size) > 0) {
      endflag = callback(vec, veclabel, &status);

      if (endflag) {
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
    }

    if (buf_size == (*cur_size) || !endflag) {
      switch (type) {
      case BUF_TRAIN: {
        std::lock_guard<std::mutex> lgtrain(readyTrainData);
        if (!endflag) {
          trainReadyFlag = DATA_END;
          cv_train.notify_all();
          free(vec);
          free(veclabel);
          return;
        } else {
          trainReadyFlag = DATA_READY;
          cv_train.notify_all();
        }

      } break;
      case BUF_VAL: {
        std::lock_guard<std::mutex> lgval(readyValData);
        if (!endflag) {
          valReadyFlag = DATA_END;
          cv_train.notify_all();
          free(vec);
          free(veclabel);
          return;
        } else {
          valReadyFlag = DATA_READY;
          cv_val.notify_all();
        }

      } break;
      case BUF_TEST: {
        std::lock_guard<std::mutex> lgtest(readyTestData);
        if (!endflag) {
          testReadyFlag = DATA_END;
          cv_test.notify_all();
          free(vec);
          free(veclabel);
          return;
        } else {
          testReadyFlag = DATA_READY;
          cv_test.notify_all();
        }

      } break;
      default:
        break;
      }
    }
  }
  free(vec);
  free(veclabel);
}

} /* namespace nntrainer */
