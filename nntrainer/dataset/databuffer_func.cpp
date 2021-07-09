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

namespace nntrainer {

int DataBufferFromCallback::init() {
  int status = ML_ERROR_NONE;

  status = DataBuffer::init();
  if (status != ML_ERROR_NONE)
    return status;

  if (callback == nullptr)
    return ML_ERROR_BAD_ADDRESS;

  samples_per_epoch = 0;

  if (buf_size > batch_size || buf_size == 0) {
    buf_size = batch_size;
  }

  return ML_ERROR_NONE;
}

int DataBufferFromCallback::setGeneratorFunc(datagen_cb func, void *user_data) {
  if (!func)
    return ML_ERROR_INVALID_PARAMETER;
  callback = func;
  this->user_data = user_data;
  initialized = true;
  return ML_ERROR_NONE;
}

void DataBufferFromCallback::updateData() {
  int status = ML_ERROR_NONE;

  setStateAndNotify(DataStatus::DATA_NOT_READY);
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
    consumer_exception_ptr = std::current_exception();
    setStateAndNotify(DataStatus::DATA_ERROR);
    return;
  }

  vec_arr[0] = vec;
  veclabel_arr[0] = veclabel;

  while (is_running) {
    endflag = false;
    setStateAndNotify(DataStatus::DATA_NOT_READY);
    if (buf_size - cur_bufsize > 0) {
      /** @todo Update to support multiple inputs later */
      status = callback(vec_arr, veclabel_arr, &endflag, user_data);
      if (endflag) {
        setStateAndNotify(DataStatus::DATA_END);
        free(vec);
        free(veclabel);
        free(vec_arr);
        free(veclabel_arr);
        return;
      }
      if (status != ML_ERROR_NONE) {
        setStateAndNotify(DataStatus::DATA_ERROR);
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
        data_q.push_back(v);
        label_q.push_back(vl);
        cur_bufsize++;
        data_lock.unlock();
      }
    }
    if (buf_size == cur_bufsize) {
      setStateAndNotify(DataStatus::DATA_READY);
    }
  }

  free(vec);
  free(veclabel);
  free(vec_arr);
  free(veclabel_arr);
}

} /* namespace nntrainer */
