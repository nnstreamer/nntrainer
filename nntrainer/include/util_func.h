/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	util_func.h
 * @date	08 April 2020
 * @brief	This is collection of math functions.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __UTIL_FUNC_H__
#define __UTIL_FUNC_H__
#ifdef __cplusplus

#include <layer.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     get the seed
 * @retVal    seed
 */
unsigned int getSeed();

/**
 * @brief     random function
 */
float random();

/**
 * @brief     sqrt function for float type
 * @param[in] x float
 */
float sqrtFloat(float x);

double sqrtDouble(double x);

/**
 * @brief     log function for float type
 * @param[in] x float
 */
float logFloat(float x);

/**
 * @brief     exp function for float type
 * @param[in] x float
 */
float exp_util(float x);

/**
 * @brief     apply padding
 * @param[in] batch batch index
 * @param[in] x input
 * @param[in] padding 2D padding size
 * @retVal Tensor output tensor with batch size is 1 for batch index
 */
Tensor zero_pad(int batch, Tensor const &in, unsigned int const *padding);

/**
 * @brief     strip padding
 * @param[in] x input
 * @param[in] padding 2D padding size
 * @retVal Tensor output tensor without padding
 */
Tensor strip_pad(Tensor const &in, unsigned int const *padding);

/**
 * @brief     rotate 180 dgree
 * @param[in] in input Tensor
 * @retVal Tensor rotated tensor (180 degree)
 */
Tensor rotate_180(Tensor in);

/**
 * @brief     Check Existance of File
 * @param[in] file path of the file to be checked
 * @returns   true if file exists, else false
 */
bool isFileExist(std::string file);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_FUNC_H__ */
