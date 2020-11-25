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
 * @file	parse_util.h
 * @date	07 May 2020
 * @brief	This is collection of parse functions.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __PARSE_UTIL_H__
#define __PARSE_UTIL_H__
#ifdef __cplusplus

#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <ml-api-common.h>

namespace nntrainer {

#define NN_RETURN_STATUS()         \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      return status;               \
    }                              \
  } while (0)

/**
 * @brief     Enumeration for input configuration file parsing
 *            0. LOSS    ( Loss Function Token )
 *            1. MODEL   ( Model Token )
 *            2. ACTI    ( Activation Token )
 *            3. WEIGHT_INIT  ( Weight Initializer Token )
 *            4. WEIGHT_REGULARIZER  ( Weight Decay Token )
 *            5. PADDING  ( Padding Token )
 *            6. POOLING  ( Pooling Token )
 *            7. UNKNOWN
 */
typedef enum {
  TOKEN_LOSS,
  TOKEN_MODEL,
  TOKEN_ACTI,
  TOKEN_WEIGHT_INIT,
  TOKEN_WEIGHT_REGULARIZER,
  TOKEN_PADDING,
  TOKEN_POOLING,
  TOKEN_UNKNOWN
} InputType;

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
 * @brief     Parsing Layer Property
 * @param[in] property string to be parsed
 * @retval    int enumerated type
 */
unsigned int parseLayerProperty(std::string property);

/**
 * @brief     Unparse Layer property to string
 * @param[in] type property type
 * @retval    string representation of the type
 */
std::string propToStr(const unsigned int type);

/**
 * @brief     Parsing Configuration Token
 * @param[in] ll string to be parsed
 * @param[in] t  Token type
 * @retval    int enumerated type
 */
unsigned int parseType(std::string ll, InputType t);

/**
 * @brief     Parsing Optimizer Property
 * @param[in] property string to be parsed
 * @retval    int enumerated type
 */
unsigned int parseOptProperty(std::string property);

/**
 * @brief     Parsing Network Property
 * @param[in] property string to be parsed
 * @retval    int enumerated type
 */
unsigned int parseNetProperty(std::string property);

/**
 * @brief     Parsing Data Buffer Property
 * @param[in] property string to be parsed
 * @retval    int enumerated type
 */
unsigned int parseDataProperty(std::string property);

/**
 * @brief     check str to be unsigned int and assign to variable to type T
 * @param[out] val assign variable
 * @param[in] str input string
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int setUint(unsigned int &val, const std::string &str);

/**
 * @brief     check str to be float and assign
 * @param[out] val assign variable
 * @param[in] str input string
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int setFloat(float &val, std::string str);

/**
 * @brief     check str to be double and assign
 * @param[out] val assign variable
 * @param[in] str input string
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int setDouble(double &val, std::string str);

/**
 * @brief     check str to be bool and assign
 * @param[out] val assign variable
 * @param[in] str input string
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int setBoolean(bool &val, std::string str);

/**
 * @brief     parse string and return key & value
 * @param[in] input_str input string to split with '='
 * @param[out] key key
 * @param[out] value value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int getKeyValue(std::string &input_str, std::string &key, std::string &value);

/**
 * @brief     join vector of int to string with delimiter ","
 * @param[in] values vector of int
 * @param[in] delimiter delimiter for the string
 * @retval    output string
 */
const char *getValues(std::vector<int> values, const char *delimiter = ",");

int getValues(int n_str, std::string str, int *value);

/**
 * @brief     split string into vector with delimiter regex
 * @param[in] str string
 * @param[in] reg regular expression to use as delimiter
 * @retval    output string vector
 */
std::vector<std::string> split(const std::string &s, const std::regex &reg);

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
 * @brief Cast insensitive string comparison
 *
 * @param a first string to compare
 * @param b second string to compare
 * @return true if string is case-insensitive equal
 * @return false if string is case-insensitive not equal
 */
bool istrequal(const std::string &a, const std::string &b);
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __PARSE_UTIL_H__ */
