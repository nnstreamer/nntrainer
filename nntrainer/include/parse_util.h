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
#include <string>
#include <vector>

namespace nntrainer {

#define NN_RETURN_STATUS()         \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      return status;               \
    }                              \
  } while (0)

/**
 * @brief     Enumeration for input configuration file parsing
 *            0. OPT     ( Optimizer Token )
 *            1. COST    ( Cost Function Token )
 *            2. MODEL   ( Model Token )
 *            3. ACTI    ( Activation Token )
 *            4. LAYER   ( Layer Token )
 *            5. WEIGHT_INIT  ( Weight Initializer Token )
 *            7. WEIGHT_DECAY  ( Weight Decay Token )
 *            8. PADDING  ( Padding Token )
 *            9. POOLING  ( Pooling Token )
 *            9. UNKNOWN
 */
typedef enum {
  TOKEN_OPT,
  TOKEN_COST,
  TOKEN_MODEL,
  TOKEN_ACTI,
  TOKEN_LAYER,
  TOKEN_WEIGHT_INIT,
  TOKEN_WEIGHT_DECAY,
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
int getKeyValue(std::string input_str, std::string &key, std::string &value);

/**
 * @brief     join vector of int to string with delimiter ","
 * @param[in] values vector of int
 * @param[in] delimiter delimiter for the string
 * @retval    output string
 */
const char *getValues(std::vector<int> values, const char *delimiter = ",");

int getValues(int n_str, std::string str, int *value);

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

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __PARSE_UTIL_H__ */
