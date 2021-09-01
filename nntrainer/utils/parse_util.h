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
#include <nntrainer_error.h>
#include <props_util.h>

namespace nntrainer {

/**
 * @brief     Enumeration for input configuration file parsing
 *            0. MODEL   ( Model Token )
 *            1. ACTI    ( Activation Token )
 *            2. WEIGHT_INIT  ( Weight Initializer Token )
 *            3. WEIGHT_REGULARIZER  ( Weight Decay Token )
 *            4. PADDING  ( Padding Token )
 *            5. POOLING  ( Pooling Token )
 *            6. UNKNOWN
 */
typedef enum {
  TOKEN_MODEL,
  TOKEN_ACTI,
  TOKEN_WEIGHT_INIT,
  TOKEN_WEIGHT_REGULARIZER,
  TOKEN_POOLING,
  TOKEN_UNKNOWN
} InputType;

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

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __PARSE_UTIL_H__ */
