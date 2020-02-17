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
 * @brief	This is Matrix class for calculation using blas library
 * @see https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "include/databuffer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

std::mutex data_lock;

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

  this->cur_train_bufsize = train_bufsize;
  this->cur_val_bufsize = val_bufsize;
  this->cur_test_bufsize = test_bufsize;

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

  for (unsigned int i = 0; i < train_bufsize; ++i) {
    std::vector<float> vec;
    std::vector<float> veclabel;
    for (unsigned int j = 0; j < input_size; ++j) {
      float data;
      train_file.read((char *)&data, sizeof(float));
      vec.push_back(data);
    }
    trainData.push_back(vec);
    for (unsigned int j = 0; j < class_num; ++j) {
      float data;
      train_file.read((char *)&data, sizeof(float));
      veclabel.push_back(data);
    }
    trainDataLabel.push_back(veclabel);
    rest_train--;
  }
  for (unsigned int i = 0; i < val_bufsize; ++i) {
    std::vector<float> vec;
    std::vector<float> veclabel;
    for (unsigned int j = 0; j < input_size; ++j) {
      float data;
      val_file.read((char *)&data, sizeof(float));
      vec.push_back(data);
    }
    valData.push_back(vec);
    for (unsigned int j = 0; j < class_num; ++j) {
      float data;
      val_file.read((char *)&data, sizeof(float));
      veclabel.push_back(data);
    }
    valDataLabel.push_back(veclabel);
    rest_val--;
  }

  for (unsigned int i = 0; i < test_bufsize; ++i) {
    std::vector<float> vec;
    std::vector<float> veclabel;
    for (unsigned int j = 0; j < input_size; ++j) {
      float data;
      test_file.read((char *)&data, sizeof(float));
      vec.push_back(data);
    }
    testData.push_back(vec);
    for (unsigned int j = 0; j < class_num; ++j) {
      float data;
      test_file.read((char *)&data, sizeof(float));
      veclabel.push_back(data);
    }
    testDataLabel.push_back(veclabel);
    rest_test--;
  }
  return true;
}

void DataBuffer::UpdateData(buffer_type type, std::ifstream &file) {
  switch (type) {
    case BUF_TRAIN: {
      std::vector<unsigned int> mark;
      mark.resize(max_train);
      for (unsigned int i = 0; i < max_train; ++i) {
        mark[i] = i;
      }
      unsigned int I;
      while (train_running || mark.size() == 0) {
        if (train_bufsize - cur_train_bufsize >= mini_batch && rest_train > mini_batch) {
          data_lock.lock();
          std::vector<int> list;
          for (unsigned int i = 0; i < mini_batch; ++i) {
            std::vector<float> vec;
            std::vector<float> veclabel;

            unsigned int id = rangeRandom(0, mark.size() - 1);
            I = mark[id];
            mark.erase(mark.begin() + id);

            list.push_back(I);
            int64_t position = (I * input_size + I * class_num) * sizeof(float);
            file.seekg(position, std::ios::beg);

            for (unsigned int j = 0; j < input_size; ++j) {
              float data;
              file.read((char *)&data, sizeof(float));
              vec.push_back(data);
            }
            trainData.push_back(vec);
            for (unsigned int j = 0; j < class_num; ++j) {
              float data;
              file.read((char *)&data, sizeof(float));
              veclabel.push_back(data);
            }
            trainDataLabel.push_back(veclabel);
            rest_train--;
          }
          cur_train_bufsize += mini_batch;
          data_lock.unlock();
        }
      }
    } break;
    case BUF_VAL: {
      std::vector<int> mark;
      mark.resize(max_val);

      memset(mark.data(), 0, sizeof(int) * max_val);

      for (unsigned int i = 0; i < max_val; ++i) {
        mark[i] = i;
      }

      int I;

      while (val_running || mark.size() == 0) {
        if (val_bufsize - cur_val_bufsize >= mini_batch && rest_val > mini_batch) {
          data_lock.lock();
          for (unsigned int i = 0; i < mini_batch; ++i) {
            std::vector<float> vec;
            std::vector<float> veclabel;

            unsigned int id = rangeRandom(0, mark.size() - 1);
            I = mark[id];
            mark.erase(mark.begin() + id);

            int64_t position = (I * input_size + I * class_num) * sizeof(float);
            file.seekg(position, std::ios::beg);

            for (unsigned int j = 0; j < input_size; ++j) {
              float data;
              file.read((char *)&data, sizeof(float));
              vec.push_back(data);
            }
            valData.push_back(vec);
            for (unsigned int j = 0; j < class_num; ++j) {
              float data;
              file.read((char *)&data, sizeof(float));
              veclabel.push_back(data);
            }
            valDataLabel.push_back(veclabel);
            rest_val--;
          }
          cur_val_bufsize += mini_batch;
          data_lock.unlock();
        }
      }
    } break;
    case BUF_TEST: {
      std::vector<int> mark;
      mark.resize(max_test);
      for (unsigned int i = 0; i < max_test; ++i) {
        mark[i] = i;
      }
      int I;

      while (test_running || mark.size() == 0) {
        if (test_bufsize - cur_test_bufsize >= mini_batch && rest_test > mini_batch) {
          data_lock.lock();
          for (unsigned int i = 0; i < mini_batch; ++i) {
            std::vector<float> vec;
            std::vector<float> veclabel;

            unsigned int id = rangeRandom(0, mark.size() - 1);
            I = mark[id];
            mark.erase(mark.begin() + id);

            int64_t position = (I * input_size + I * class_num) * sizeof(float);
            file.seekg(position, std::ios::beg);

            for (unsigned int j = 0; j < input_size; ++j) {
              float data;
              file.read((char *)&data, sizeof(float));
              vec.push_back(data);
            }
            testData.push_back(vec);
            for (unsigned int j = 0; j < class_num; ++j) {
              float data;
              file.read((char *)&data, sizeof(float));
              veclabel.push_back(data);
            }
            testDataLabel.push_back(veclabel);
            rest_test--;
          }
          cur_test_bufsize += mini_batch;
          data_lock.unlock();
        }
      }
    } break;
    default:
      break;
  }
}

void DataBuffer::run(buffer_type type, std::ifstream &file) {
  switch (type) {
    case BUF_TRAIN:
      this->train_thread = std::thread(&DataBuffer::UpdateData, this, BUF_TRAIN, std::ref(file));
      this->train_thread.detach();
      break;
    case BUF_VAL:
      this->val_thread = std::thread(&DataBuffer::UpdateData, this, BUF_VAL, std::ref(file));
      this->val_thread.detach();
      break;
    case BUF_TEST:
      this->test_thread = std::thread(&DataBuffer::UpdateData, this, BUF_TEST, std::ref(file));
      this->test_thread.detach();
      break;
    default:
      break;
  }
}

bool DataBuffer::getStatus(buffer_type type) {
  bool ret = true;
  switch (type) {
    case BUF_TRAIN:
      if (trainData.size() < mini_batch)
        ret = false;
      break;
    case BUF_VAL:
      if (valData.size() < mini_batch)
        ret = false;
      break;
    case BUF_TEST:
      if (testData.size() < mini_batch)
        ret = false;
      break;
    default:
      break;
  }
  return ret;
}

void DataBuffer::clear(buffer_type type, std::ifstream &file) {
  switch (type) {
    case BUF_TRAIN: {
      train_running = false;
      this->trainData.clear();
      this->trainDataLabel.clear();
      this->cur_train_bufsize = this->train_bufsize;
      this->rest_train = max_train;

      this->train_running = true;
      file.clear();
      file.seekg(0, std::ios::beg);

      for (unsigned int i = 0; i < train_bufsize; ++i) {
        std::vector<float> vec;
        std::vector<float> veclabel;
        for (unsigned int j = 0; j < input_size; ++j) {
          float data;
          file.read((char *)&data, sizeof(float));
          vec.push_back(data);
        }
        trainData.push_back(vec);
        for (unsigned int j = 0; j < class_num; ++j) {
          float data;
          file.read((char *)&data, sizeof(float));
          veclabel.push_back(data);
        }
        trainDataLabel.push_back(veclabel);
        rest_train--;
      }
    } break;
    case BUF_VAL: {
      val_running = false;
      this->valData.clear();
      this->cur_val_bufsize = this->val_bufsize;
      this->rest_val = max_val;

      this->val_running = true;
      file.clear();
      file.seekg(0, std::ios::beg);

      for (unsigned int i = 0; i < val_bufsize; ++i) {
        std::vector<float> vec;
        std::vector<float> veclabel;
        for (unsigned int j = 0; j < input_size; ++j) {
          float data;
          file.read((char *)&data, sizeof(float));
          vec.push_back(data);
        }
        valData.push_back(vec);
        for (unsigned int j = 0; j < class_num; ++j) {
          float data;
          file.read((char *)&data, sizeof(float));
          veclabel.push_back(data);
        }
        valDataLabel.push_back(veclabel);
        rest_val--;
      }

    } break;
    case BUF_TEST: {
      test_running = false;
      this->testData.clear();
      this->cur_test_bufsize = this->test_bufsize;
      this->rest_test = max_test;

      this->test_running = true;

      file.clear();
      file.seekg(0, std::ios::beg);

      for (unsigned int i = 0; i < test_bufsize; ++i) {
        std::vector<float> vec;
        std::vector<float> veclabel;
        for (unsigned int j = 0; j < input_size; ++j) {
          float data;
          file.read((char *)&data, sizeof(float));
          vec.push_back(data);
        }
        testData.push_back(vec);
        for (unsigned int j = 0; j < class_num; ++j) {
          float data;
          file.read((char *)&data, sizeof(float));
          veclabel.push_back(data);
        }
        testDataLabel.push_back(veclabel);
        rest_test--;
      }
    } break;
    default:
      break;
  }
}

bool DataBuffer::getDatafromBuffer(buffer_type type, std::vector<std::vector<std::vector<float>>> &outVec,
                                   std::vector<std::vector<std::vector<float>>> &outLabel, unsigned int batch,
                                   unsigned int width, unsigned int height, unsigned int c_num) {
  int nomI;
  unsigned int J, i, j, k;

  switch (type) {
    case BUF_TRAIN: {
      std::vector<int> list;

      if (!getStatus(BUF_TRAIN))
        return false;
      data_lock.lock();
      for (k = 0; k < batch; ++k) {
        nomI = rangeRandom(0, trainData.size() - 1);
        std::vector<std::vector<float>> v_height;
        for (j = 0; j < height; ++j) {
          J = j * width;
          std::vector<float> v_width;
          for (i = 0; i < width; ++i) {
            v_width.push_back(trainData[nomI][J + i]);
          }
          v_height.push_back(v_width);
        }

        list.push_back(nomI);
        outVec.push_back(v_height);
        outLabel.push_back({trainDataLabel[nomI]});
      }
      for (i = 0; i < batch; ++i) {
        trainData.erase(trainData.begin() + list[i]);
        trainDataLabel.erase(trainDataLabel.begin() + list[i]);
        cur_train_bufsize--;
      }
    } break;
    case BUF_VAL: {
      std::vector<int> list;
      if (!getStatus(BUF_VAL))
        return false;
      data_lock.lock();
      for (k = 0; k < batch; ++k) {
        nomI = rangeRandom(0, valData.size() - 1);
        std::vector<std::vector<float>> v_height;
        for (j = 0; j < height; ++j) {
          J = j * width;
          std::vector<float> v_width;
          for (i = 0; i < width; ++i) {
            v_width.push_back(valData[nomI][J + i]);
          }
          v_height.push_back(v_width);
        }

        list.push_back(nomI);
        outVec.push_back(v_height);
        outLabel.push_back({valDataLabel[nomI]});
      }
      for (i = 0; i < batch; ++i) {
        valData.erase(valData.begin() + list[i]);
        valDataLabel.erase(valDataLabel.begin() + list[i]);
        cur_val_bufsize--;
      }
    } break;
    case BUF_TEST: {
      std::vector<int> list;
      if (!getStatus(BUF_TEST))
        return false;
      data_lock.lock();
      for (k = 0; k < batch; ++k) {
        nomI = rangeRandom(0, testData.size() - 1);
        std::vector<std::vector<float>> v_height;
        for (j = 0; j < height; ++j) {
          J = j * width;
          std::vector<float> v_width;
          for (i = 0; i < width; ++i) {
            v_width.push_back(testData[nomI][J + i]);
          }
          v_height.push_back(v_width);
        }

        list.push_back(nomI);
        outVec.push_back(v_height);
        outLabel.push_back({testDataLabel[nomI]});
      }
      for (i = 0; i < batch; ++i) {
        testData.erase(testData.begin() + list[i]);
        testDataLabel.erase(testDataLabel.begin() + list[i]);
        cur_test_bufsize--;
      }
    }
    break;
  default:
    return false;
    break;
  }
  data_lock.unlock();
  return true;
}
