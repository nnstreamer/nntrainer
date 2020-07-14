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

#include <atomic>
#include <databuffer.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

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
  int init();

  /**
   * @brief     set function pointer for each type
   * @param[in] type Buffer Type
   * @param[in] call back function pointer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setFunc(BufferType type,
              std::function<bool(float *, float *, int *)> func);

  /**
   * @brief     Update Data Buffer ( it is for child thread )
   * @param[in] BufferType training, validation, test
   * @retval    void
   */
  void updateData(BufferType type);

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
  std::function<bool(float *, float *, int *)> callback_train;
  std::function<bool(float *, float *, int *)> callback_val;
  std::function<bool(float *, float *, int *)> callback_test;
};
} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_FUNC_H__ */
