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

  if (initialized && samples_per_epoch < batch_size) {
    samples_per_epoch = batch_size;
  }

  this->remaining_samples_per_epoch = samples_per_epoch;

  if (initialized && samples_per_epoch < buf_size) {
    ml_logw("Warning: Total number of train is less than train buffer size. "
            "Train buffer size is set as total number of train");
    buf_size = batch_size;
  }

  return ML_ERROR_NONE;
}

void DataBufferFromDataFile::updateData() {
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
  /// @todo check if file is good

  setStateAndNotify(DataStatus::DATA_NOT_READY);

  unsigned int I;
  std::vector<unsigned int> mark;
  mark.resize(samples_per_epoch);
  file.clear();
  file.seekg(0, std::ios_base::end);
  uint64_t file_length = file.tellg();

  for (unsigned int i = 0; i < samples_per_epoch; ++i) {
    mark[i] = i;
  }

  while (is_running) {
    if (mark.size() == 0) {
      setStateAndNotify(DataStatus::DATA_END);
      break;
    }

    if (buf_size - cur_bufsize > 0 && remaining_samples_per_epoch > 0) {
      std::vector<float> vec;
      std::vector<float> veclabel;

      unsigned int id = rangeRandom(0, mark.size() - 1);
      I = mark[id];

      try {
        if (I > samples_per_epoch) {
          throw std::runtime_error(
            "Error: Test case id cannot exceed maximum number of test");
        }
      } catch (...) {
        consumer_exception_ptr = std::current_exception();
        setStateAndNotify(DataStatus::DATA_ERROR);
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
        consumer_exception_ptr = std::current_exception();
        setStateAndNotify(DataStatus::DATA_ERROR);
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
      data_q.push_back(vec);
      label_q.push_back(veclabel);
      remaining_samples_per_epoch--;
      cur_bufsize++;
      data_lock.unlock();
    }
    if (buf_size == cur_bufsize) {
      setStateAndNotify(DataStatus::DATA_READY);
    }
  }

  file.close();
}

int DataBufferFromDataFile::setDataFile(const std::string &path) {
  int status = ML_ERROR_NONE;
  std::ifstream data_file(path.c_str());

  initialized = true;
  if (!data_file.good()) {
    ml_loge("Error: Cannot open data file, Datafile is necessary for training");
    initialized = false;
    return ML_ERROR_INVALID_PARAMETER;
  }
  file_name = path;

  ml_logd("datafile has set. path: %s", path.c_str());

  return status;
}

int DataBufferFromDataFile::setFeatureSize(const TensorDim &tdim) {
  int status = ML_ERROR_NONE;
  long file_size = 0;

  status = DataBuffer::setFeatureSize(tdim);
  if (status != ML_ERROR_NONE)
    return status;

  if (initialized) {
    file_size = getFileSize(file_name);
    samples_per_epoch = static_cast<unsigned int>(
      file_size /
      (class_num * sizeof(int) + input_dim.getFeatureLen() * sizeof(float)));
    if (samples_per_epoch < batch_size) {
      ml_logw("Warning: number of training data is smaller than batch size");
    }
  } else {
    samples_per_epoch = 0;
  }

  return status;
}

} /* namespace nntrainer */
