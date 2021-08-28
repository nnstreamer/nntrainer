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

#include <cstring>
#include <sstream>

#include <nntrainer_error.h>
#include <tensor.h>
namespace nntrainer {

#define NN_RETURN_STATUS()         \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      return status;               \
    }                              \
  } while (0)

/**
 * @brief convert integer based status to throw
 *
 * @param status status to throw
 */
inline void throw_status(int status) {
  switch (status) {
  case ML_ERROR_NONE:
    break;
  case ML_ERROR_INVALID_PARAMETER:
    throw std::invalid_argument("invalid argument from c style throw");
  case ML_ERROR_OUT_OF_MEMORY:
    throw std::bad_alloc();
  case ML_ERROR_TIMED_OUT:
    throw std::runtime_error("Timed out from c style throw");
  case ML_ERROR_PERMISSION_DENIED:
    throw std::runtime_error("permission denied from c style throw");
  case ML_ERROR_UNKNOWN:
  default:
    throw std::runtime_error("unknown error from c style throw");
  }
}

/**
 * @brief     get the seed
 * @return    seed
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

/**
 * @brief    sqrt function for dobuld type
 *
 * @param x value to take sqrt
 * @return double return value
 */
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

/**
 * @brief check if string ends with @a suffix
 *
 * @param target string to cehck
 * @param suffix check if string ends with @a suffix
 * @retval true @a target ends with @a suffix
 * @retval false @a target does not ends with @a suffix
 */
bool endswith(const std::string &target, const std::string &suffix);

/**
 * @brief     print instance info. as <Type at (address)>
 * @param[in] std::ostream &out, T&& t
 * @param[in] t pointer to the instance
 */
template <typename T,
          typename std::enable_if_t<std::is_pointer<T>::value, T> * = nullptr>
void printInstance(std::ostream &out, const T &t) {
  out << '<' << typeid(*t).name() << " at " << t << '>' << std::endl;
}

/**
 * @brief creat a stream, and if !stream.good() throw appropriate error code
 * depending on @c errno
 *
 * @tparam T return type
 * @param path path
 * @param mode mode to open path
 * @return T created stream
 */
template <typename T>
T checkedOpenStream(const std::string &path, std::ios_base::openmode mode) {
  T model_file(path, mode);
  if (!model_file.good()) {
    std::stringstream ss;
    ss << "[parseutil] requested file not opened, file path: " << path
       << " reason: " << std::strerror(errno);
    if (errno == EPERM || errno == EACCES) {
      throw nntrainer::exception::permission_denied(ss.str().c_str());
    } else {
      throw std::invalid_argument(ss.str().c_str());
    }
  }

  return model_file;
}

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_FUNC_H__ */
