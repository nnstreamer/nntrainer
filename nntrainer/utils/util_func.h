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
 * @param[out] padded output
 */
void zero_pad(int batch, Tensor const &in, unsigned int const *padding,
              Tensor &output);

/**
 * @brief     strip padding
 * @param[in] x input
 * @param[in] padding 2D padding size
 * @param[in] output output tensor
 * @param[in] batch batch index
 */
void strip_pad(Tensor const &in, unsigned int const *padding, Tensor &output,
               unsigned int batch);

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

constexpr const char *default_error_msg =
  "[util::checkeFile] file operation failed";

/**
 * @brief same as file.read except it checks if fail to read the file
 *
 * @param file file to read
 * @param array char * array
 * @param size size of the array
 * @param error_msg error msg to print when operation fail
 * @throw std::runtime_error if file.fail() is true after read.
 */
void checkedRead(std::ifstream &file, char *array, std::streamsize size,
                 const char *error_msg = default_error_msg);

/**
 * @brief same as file.write except it checks if fail to write the file
 *
 * @param file file to write
 * @param array char * array
 * @param size size of the array
 * @param error_msg error msg to print when operation fail
 * @throw std::runtime_error if file.fail() is true after write.
 */
void checkedWrite(std::ofstream &file, const char *array, std::streamsize size,
                  const char *error_msg = default_error_msg);
/**
 * @brief read string from a binary file
 *
 * @param file file to input
 * @return std::string result string
 */
std::string readString(std::ifstream &file,
                       const char *error_msg = default_error_msg);

/**
 * @brief write string to a binary file
 *
 * @param file file to write
 * @param str target string to write
 */
void writeString(std::ofstream &file, const std::string &str,
                 const char *error_msg = default_error_msg);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_FUNC_H__ */
