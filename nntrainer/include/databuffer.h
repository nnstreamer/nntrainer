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
 * @brief	This is buffer object to handle big data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __DATABUFFER_H__
#define __DATABUFFER_H__
#ifdef __cplusplus

#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace nntrainer {

/**
 * @brief     Enumeration of buffer type
 *            0. BUF_TRAIN ( Buffer for training )
 *            1. BUF_VAL ( Buffer for validation )
 *            2. BUF_TEST ( Buffer for test )
 *            3. Unknown
 */
typedef enum { BUF_TRAIN, BUF_VAL, BUF_TEST, BUFF_UNKNOWN } BufferType;

/**
 * @brief     Enumeration of data type
 *            0. DATA_TRAIN ( Data for training )
 *            1. DATA_VAL ( Data for validation )
 *            2. DATA_TEST ( Data for test )
 *            3. DATA_LABEL ( Data for test )
 *            3. Unknown
 */
typedef enum {
  DATA_TRAIN,
  DATA_VAL,
  DATA_TEST,
  DATA_LABEL,
  DATA_UNKNOWN
} DataType;

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
  DataBuffer()
    : train_running(), val_running(), test_running(), train_thread(),
      val_thread(), test_thread(){};

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
  bool init(int mini_batch, unsigned int train_bufsize,
            unsigned int val_bufsize, unsigned int test_bufsize,
            std::ifstream &train_file, std::ifstream &val_file,
            std::ifstream &test_file, unsigned int max_train,
            unsigned int max_val, unsigned int max_test, unsigned int in_size,
            unsigned int c_num);

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  void updateData(BufferType type, std::ifstream &file);

  /**
   * @brief     function for thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  void run(BufferType type, std::ifstream &file);

  /**
   * @brief     clear thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  void clear(BufferType type, std::ifstream &file);

  /**
   * @brief     get Status of Buffer. if number of rest data
   *            is samller than minibatch, the return false
   * @param[in] BufferType training, validation, test
   * @retval    true/false
   */
  bool getStatus(BufferType type);

  /**
   * @brief     get Data from Data Buffer
   * @param[in] BufferType training, validation, test
   * @param[in] outVec feature data ( minibatch size )
   * @param[in] outLabel label data ( minibatch size )
   * @param[in] batch size of batch
   * @param[in] width width
   * @param[in] height height
   * @param[in] c_num number of class
   * @retval    true/false
   */
  bool getDataFromBuffer(
    BufferType type, std::vector<std::vector<std::vector<float>>> &out_vec,
    std::vector<std::vector<std::vector<float>>> &out_label, unsigned int batch,
    unsigned int width, unsigned int height, unsigned int c_num);

  /**
   * @brief     set train data file name
   * @param[in] path file path
   * @param[in] type data type : DATA_TRAIN, DATA_VAL, DATA_TEST
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataFile(std::string path, DataType type);

  /**
   * @brief     set number of class
   * @param[in] number of class
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setClassNum(unsigned int n);

private:
  std::vector<std::vector<float>> train_data;
  std::vector<std::vector<float>> train_data_label;
  std::vector<std::vector<float>> val_data;
  std::vector<std::vector<float>> val_data_label;
  std::vector<std::vector<float>> test_data;
  std::vector<std::vector<float>> test_data_label;

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

  std::string train_file;
  std::string val_file;
  std::string test_file;
  std::vector<std::string> labels;
};
} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_H__ */
