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
 * @file	databuffer_file.h
 * @date	27 April 2020
 * @brief	This is buffer object take data from raw files
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __DATABUFFER_FILE_H__
#define __DATABUFFER_FILE_H__
#ifdef __cplusplus

#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include <databuffer.h>

namespace nntrainer {

/**
 * @class   DataBufferFromDataFile Data Buffer from Raw Data File
 * @brief   Data Buffer from reading raw data
 */
class DataBufferFromDataFile : public DataBuffer {

public:
  /**
   * @brief     Constructor
   */
  DataBufferFromDataFile() : DataBuffer(DataBufferType::FILE) {}

  /**
   * @brief     Destructor
   */
  ~DataBufferFromDataFile() = default;

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
   * @param[in] Input Tensor Dimension.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setFeatureSize(TensorDim indim);

  /**
   * @brief     set property
   * @param[in] type type of property
   * @param[in] value string value of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(const PropertyType type, std::string &value);

private:
  /**
   * @brief     raw data file names
   */
  std::string train_name;
  std::string val_name;
  std::string test_name;
};

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_FILE_H__ */
