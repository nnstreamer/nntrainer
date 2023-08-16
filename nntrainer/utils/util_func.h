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
#include <regex>
#include <sstream>

#include <nntrainer_error.h>
#include <random>

// /**
//  * @brief     get the seed
//  * @return    seed
//  */
// unsigned int getSeed() { return 0; }

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

static auto rng = [] {
  std::mt19937 rng;
  // rng.seed(getSeed());
  rng.seed(0);
  return rng;
}();

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
template <typename T = float> T logFloat(T x) {
  return static_cast<T>(log(x + 1.0e-20));
}

/**
 * @brief     exp function for float type
 * @param[in] x float
 */
template <typename T = float> T exp_util(T x) { return static_cast<T>(exp(x)); }

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
void checkedWrite(std::ostream &file, const char *array, std::streamsize size,
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
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::stringstream ss;
    ss << "[parseutil] requested file not opened, file path: " << path
       << " reason: " << strerror_r(errno, error_buf, error_buflen);
    if (errno == EPERM || errno == EACCES) {
      throw nntrainer::exception::permission_denied(ss.str().c_str());
    } else {
      throw std::invalid_argument(ss.str().c_str());
    }
  }

  return model_file;
}

/**
 * @brief     parse string and return key & value
 * @param[in] input_str input string to split with '='
 * @param[out] key key
 * @param[out] value value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int getKeyValue(const std::string &input_str, std::string &key,
                std::string &value);

/**
 * @brief     parse string and stored to int
 * @param[in] n_str number of data
 * @param[in] str string to parse
 * @param[in] value int value to stored
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int getValues(int n_str, std::string str, int *value);

/**
 * @brief     split string into vector with delimiter regex
 * @param[in] str string
 * @param[in] reg regular expression to use as delimiter
 * @retval    output string vector
 */
std::vector<std::string> split(const std::string &s, const std::regex &reg);

/**
 * @brief Cast insensitive string comparison
 *
 * @param a first string to compare
 * @param b second string to compare
 * @retval true if string is case-insensitive equal
 * @retval false if string is case-insensitive not equal
 */
bool istrequal(const std::string &a, const std::string &b);

/**
 * @brief Perform INT_LOGICAL_AND operation on enum class value
 *
 * @param e1 enum value
 * @param e2 enum value
 *
 * @return enum value after performing logical AND operation
 */
template <typename T, typename C = int>
bool enum_class_logical_and(T e1, T e2) {
  C i1 = static_cast<int>(e1);
  C i2 = static_cast<int>(e2);

  return (i1 & i2) != 0;
}

/**
 * @brief Perform INT_OR operation on enum class value
 *
 * @param e1 enum value
 * @param e2 enum value
 *
 * @return enum value after performing OR operation
 */
template <typename T, typename C = int> T enum_class_or(T e1, T e2) {
  C i1 = static_cast<int>(e1);
  C i2 = static_cast<int>(e2);

  return static_cast<T>(i1 | i2);
}

/**
 * @brief Convert a relative path into an absolute path.
 *
 * @param name relative path
 * @param resolved variable to store the result value.
 *
 * @return absolute path
 */
char *getRealpath(const char *name, char *resolved);

/**
 * @brief Get local time in tm struct format
 *
 * @param tp variable to store the result values
 *
 * @return tm struct
 */
tm *getLocaltime(tm *tp);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_FUNC_H__ */
