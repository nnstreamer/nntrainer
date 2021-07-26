// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	props_util.h
 * @date	26 July 2021
 * @brief	This is collection of utility function for properties.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __PROPS_UTIL_H__
#define __PROPS_UTIL_H__
#ifdef __cplusplus

#include <regex>
#include <string>
#include <vector>

namespace nntrainer {

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
int getKeyValue(const std::string &input_str, std::string &key,
                std::string &value);

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
 * @brief Cast insensitive string comparison
 *
 * @param a first string to compare
 * @param b second string to compare
 * @retval true if string is case-insensitive equal
 * @retval false if string is case-insensitive not equal
 */
bool istrequal(const std::string &a, const std::string &b);
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __PARSE_UTIL_H__ */
