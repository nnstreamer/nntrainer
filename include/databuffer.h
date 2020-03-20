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
 * @file	databuffer.h
 * @date	04 December 2019
 * @brief	This is Matrix class for calculation using blas library.
 * @see https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __DATABUFFER_H__
#define __DATABUFFER_H__

#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

/**
 * @brief     Enumeration of buffer type
 *            0. BUF_TRAIN ( Buffer for training )
 *            1. BUF_VAL ( Buffer for validation )
 *            2. BUF_TEST ( Buffer for test )
 *            3. Unknown
 */
typedef enum { BUF_TRAIN, BUF_VAL, BUF_TEST, BUFF_UNKNOWN } buffer_type;

/**
 * @class   DataBuffer Data Buffers
 * @brief   Data Buffer for read and manage data
 */
class DataBuffer {
 public:
  /**
   * @brief     Create Buffer
   * @retval    DataBuffer
   */
  DataBuffer() : train_running(), val_running(), test_running(), train_thread(), val_thread(), test_thread(){};

  /**
   * @brief     Create Buffer
   * @param[in] train_bufsize size buffer
   * @param[in] val_bufsize size buffer
   * @param[in] test_bufsize size buffer
   * @retval    DataBuffer
   */
  DataBuffer(int train_bufsize, int val_bufsize, int test_bufsize);

  /**
   * @brief     Initialize Buffer
   * @param[in] mini_batch size of minibatch
   * @param[in] train_bufsize size of training buffer
   * @param[in] val_bufsize size of validation buffer
   * @param[in] test_bufsize size of test buffer
   * @param[in] train_file input file stream for training
   * @param[in] val_file input file stream for validataion
   * @param[in] test_file input file stream for test
   * @param[in] max_train maximum number of traing data
   * @param[in] max_val maximum number of validation data
   * @param[in] max_test maximum number of test data
   * @param[in] in_size input size
   * @param[in] c_num number of class
   * @retval    true / false
   */
  bool init(int mini_batch, unsigned int train_bufsize, unsigned int val_bufsize, unsigned int test_bufsize,
            std::ifstream &train_file, std::ifstream &val_file, std::ifstream &test_file, unsigned int max_train,
            unsigned int max_val, unsigned int max_test, unsigned int in_size, unsigned int c_num);

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] buffer_type training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  void UpdateData(buffer_type type, std::ifstream &file);

  /**
   * @brief     function for thread ( training, validation, test )
   * @param[in] buffer_type training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  void run(buffer_type type, std::ifstream &file);

  /**
   * @brief     clear thread ( training, validation, test )
   * @param[in] buffer_type training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  void clear(buffer_type type, std::ifstream &file);

  /**
   * @brief     get Status of Buffer. if number of rest data
   *            is samller than minibatch, the return false
   * @param[in] buffer_type training, validation, test
   * @retval    true/false
   */
  bool getStatus(buffer_type type);

  /**
   * @brief     get Data from Data Buffer
   * @param[in] buffer_type training, validation, test
   * @param[in] outVec feature data ( minibatch size )
   * @param[in] outLabel label data ( minibatch size )
   * @param[in] batch size of batch
   * @param[in] width width
   * @param[in] height height
   * @param[in] c_num number of class
   * @retval    true/false
   */
  bool getDatafromBuffer(buffer_type type, std::vector<std::vector<std::vector<float>>> &outVec,
                         std::vector<std::vector<std::vector<float>>> &outLabel, unsigned int batch, unsigned int width,
                         unsigned int height, unsigned int c_num);

 private:
  std::vector<std::vector<float>> trainData;
  std::vector<std::vector<float>> trainDataLabel;
  std::vector<std::vector<float>> valData;
  std::vector<std::vector<float>> valDataLabel;
  std::vector<std::vector<float>> testData;
  std::vector<std::vector<float>> testDataLabel;

  unsigned int input_size;
  unsigned int class_num;

  unsigned int cur_train_bufsize;
  unsigned int cur_val_bufsize;
  unsigned int cur_test_bufsize;

  unsigned int train_bufsize;
  unsigned int val_bufsize;
  unsigned int test_bufsize;

  unsigned int max_train;
  unsigned int max_val;
  unsigned int max_test;

  unsigned int rest_train;
  unsigned int rest_val;
  unsigned int rest_test;

  unsigned int mini_batch;

  bool train_running;
  bool val_running;
  bool test_running;

  std::vector<unsigned int> train_mark;
  std::vector<unsigned int> val_mark;
  std::vector<unsigned int> test_mark;

  std::thread train_thread;
  std::thread val_thread;
  std::thread test_thread;
};

#endif
