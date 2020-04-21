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

/*
 * @brief Number of Data Set
 */
#define NBUFTYPE 4

#define SET_VALIDATION(val)                                              \
  do {                                                                   \
    for (DataType i = DATA_TRAIN; i < DATA_UNKNOWN; i = DataType(i + 1)) \
      validation[i] = val;                                               \
  } while (0)

namespace nntrainer {

/**
 * @brief     Enumeration of buffer type
 *            0. BUF_TRAIN ( Buffer for training )
 *            1. BUF_VAL ( Buffer for validation )
 *            2. BUF_TEST ( Buffer for test )
 *            3. BUF_UNKNOWN
 */
typedef enum { BUF_TRAIN, BUF_VAL, BUF_TEST, BUF_UNKNOWN } BufferType;

/**
 * @brief     Enumeration of data type
 *            0. DATA_TRAIN ( Data for training )
 *            1. DATA_VAL ( Data for validation )
 *            2. DATA_TEST ( Data for test )
 *            3. DATA_LABEL ( Data for test )
 *            4. DATA_UNKNOWN
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
      val_thread(), test_thread() {
    SET_VALIDATION(true);
    input_size = 0;
    class_num = 0;
  };

  /**
   * @brief     Destructor
   */
  virtual ~DataBuffer(){};

  /**
   * @brief     Initialize Buffer with data buffer private variables
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int init() = 0;

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @param[in] file input file stream
   * @retval    void
   */
  virtual void updateData(BufferType type) = 0;

  /**
   * @brief     function for thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  virtual void run(BufferType type);

  /**
   * @brief     clear thread ( training, validation, test )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  virtual void clear(BufferType type);

  /**
   * @brief     get Status of Buffer. if number of rest data
   *            is samller than minibatch, the return false
   * @param[in] BufferType training, validation, test
   * @retval    true/false
   */
  bool getStatus(BufferType type);

  /**
   * @brief     get Data from Data Buffer using databuffer param
   * @param[in] BufferType training, validation, test
   * @param[in] outVec feature data ( minibatch size )
   * @param[in] outLabel label data ( minibatch size )
   * @retval    true/false
   */
  virtual bool
  getDataFromBuffer(BufferType type,
                    std::vector<std::vector<std::vector<float>>> &out_vec,
                    std::vector<std::vector<std::vector<float>>> &out_label);

  /**
   * @brief     set number of class
   * @param[in] number of class
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setClassNum(unsigned int n);

  /**
   * @brief     set buffer size
   * @param[in] buffer size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setBufSize(unsigned int n);

  /**
   * @brief     set mini batch size
   * @param[in] mini batch size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setMiniBatch(unsigned int n);

  /**
   * @brief     set feature size
   * @param[in] feature batch size. It is equal to input layer's hidden size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setFeatureSize(unsigned int n);

  /**
   * @brief     set feature size
   * @retval max_train
   */
  unsigned int getMaxTrain() { return max_train; }

  /**
   * @brief     set feature size
   * @retval max_val
   */
  unsigned int getMaxVal() { return max_val; }

  /**
   * @brief     set feature size
   * @retval max_test
   */
  unsigned int getMaxTest() { return max_test; }

  /**
   * @brief     Display Progress
   * @param[in] count calculated set ( mini_batch size )
   * @param[in] type buffer type ( BUF_TRAIN, BUF_VAL, BUF_TEST )
   * @retval void
   */
  void displayProgress(const int count, BufferType type, float loss);

  /**
   * @brief     return validation of data set
   * @retval validation
   */
  bool *getValidation() { return validation; }

protected:
  /**
   * @brief     Data Queues for each data set
   */
  std::vector<std::vector<float>> train_data;
  std::vector<std::vector<float>> train_data_label;
  std::vector<std::vector<float>> val_data;
  std::vector<std::vector<float>> val_data_label;
  std::vector<std::vector<float>> test_data;
  std::vector<std::vector<float>> test_data_label;

  /**
   * @brief     feature size
   */
  unsigned int input_size;

  /**
   * @brief     number of class
   */
  unsigned int class_num;

  /**
   * @brief     number of remain data for each data queue
   */
  unsigned int cur_train_bufsize;
  unsigned int cur_val_bufsize;
  unsigned int cur_test_bufsize;

  /**
   * @brief     queue size for each data set
   */
  unsigned int bufsize;

  unsigned int max_train;
  unsigned int max_val;
  unsigned int max_test;

  /**
   * @brief     remain data set size
   */
  unsigned int rest_train;
  unsigned int rest_val;
  unsigned int rest_test;

  /**
   * @brief     mini batch size
   */
  unsigned int mini_batch;

  /**
   * @brief     flags to check status
   */
  bool train_running;
  bool val_running;
  bool test_running;

  /**
   * @brief     ids to check duplication
   */
  std::vector<unsigned int> train_mark;
  std::vector<unsigned int> val_mark;
  std::vector<unsigned int> test_mark;

  /**
   * @brief     threads to generate data for queue
   */
  std::thread train_thread;
  std::thread val_thread;
  std::thread test_thread;

  std::vector<std::string> labels;
  bool validation[NBUFTYPE];
};

/**
 * @class   DataBufferFromDataFile Data Buffer from Raw Data File
 * @brief   Data Buffer from reading raw data
 */
class DataBufferFromDataFile : public DataBuffer {

public:
  /**
   * @brief     Constructor
   */
  DataBufferFromDataFile(){};

  /**
   * @brief     Destructor
   */
  ~DataBufferFromDataFile(){};

  /**
   * @brief     Initialize Buffer with data buffer private variables
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int init();

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  void updateData(BufferType type);

  /**
   * @brief     set train data file name
   * @param[in] path file path
   * @param[in] type data type : DATA_TRAIN, DATA_VAL, DATA_TEST
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataFile(std::string path, DataType type);

  /**
   * @brief     set feature size
   * @param[in] feature batch size. It is equal to input layer's hidden size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setFeatureSize(unsigned int n);

private:
  /**
   * @brief     raw data file names
   */
  std::string train_name;
  std::string val_name;
  std::string test_name;
};

/**
 * @class   DataBufferFromCallback Data Buffer from callback given by user
 * @brief   Data Buffer from callback function
 * NYI
 */
class DataBufferFromCallback : public DataBuffer {
public:
  /**
   * @brief     Constructor
   */
  DataBufferFromCallback(){};

  /**
   * @brief     Destructor
   */
  ~DataBufferFromCallback(){};

  /**
   * @brief     Initialize Buffer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int init() {
    /* NYI */
    return 0;
  };

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  void updateData(BufferType type){
    /* NYI */
  };

private:
  /**
   * @brief     Callback function given by user
   */
  int (*callback)(BufferType, std::vector<std::vector<std::vector<float>>>,
                  std::vector<std::vector<std::vector<float>>>);
};
} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_H__ */
