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

#ifndef __DATABUFFER_FUNC_H__
#define __DATABUFFER_FUNC_H__
#ifdef __cplusplus

#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <databuffer.h>
#include <nntrainer-api-common.h>

namespace nntrainer {

/**
 * @class   DataBufferFromCallback Data Buffer from callback given by user
 * @brief   Data Buffer from callback function
 */
class DataBufferFromCallback : public DataBuffer {
public:
  /**
   * @brief     Constructor
   */
  DataBufferFromCallback() : DataBuffer(DataBufferType::GENERATOR) {}

  /**
   * @brief     Destructor
   */
  ~DataBufferFromCallback() = default;

  /**
   * @brief     Initialize Buffer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int init();

  /**
   * @brief     set function pointer for each type
   * @param[in] type Buffer Type
   * @param[in] call back function pointer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setGeneratorFunc(BufferType type, datagen_cb func);

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  void updateData(BufferType type);

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
   *
   * @brief Callback function to get user specific data
   * @param[in] X data  3D float vector type
   * @param[in] Y label 3D float vector type
   * @param[out] status status for error handle
   * @retval true / false generate all data for this epoch
   *
   */
  datagen_cb callback_train;
  datagen_cb callback_val;
  datagen_cb callback_test;
};
} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_FUNC_H__ */
