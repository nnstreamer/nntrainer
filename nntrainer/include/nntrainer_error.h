/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */
/**
 * @file nntrainer_error.h
 * @date 03 April 2020
 * @brief NNTrainer Error Codes
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNTRAINER_ERROR_H__
#define __NNTRAINER_ERROR_H__

#include <ml-api-common.h>
#if defined(__TIZEN__)
#include <tizen_error.h>
#define ML_ERROR_CANNOT_ASSIGN_ADDRESS TIZEN_ERROR_CANNOT_ASSIGN_ADDRESS
#define ML_ERROR_BAD_ADDRESS TIZEN_ERROR_BAD_ADDRESS
#define ML_ERROR_RESULT_OUT_OF_RANGE TIZEN_ERROR_RESULT_OUT_OF_RANGE
#else
#include <errno.h>
#define ML_ERROR_CANNOT_ASSIGN_ADDRESS (-EADDRNOTAVAIL)
#define ML_ERROR_BAD_ADDRESS (-EFAULT)
#define ML_ERROR_RESULT_OUT_OF_RANGE (-ERANGE)
#endif

#include <stdexcept>
namespace nntrainer {

/// @note underscore_case is used for ::exception to keep in accordance with
/// std::exception
namespace exception {

/**
 * @brief derived class of invalid argument to represent specific functionality
 * not supported
 * @note this could be either intended or not yet implemented
 */
struct not_supported : public std::invalid_argument {
  using invalid_argument::invalid_argument;
};

} // namespace exception

} // namespace nntrainer

#endif /* __NNTRAINER_ERROR_H__ */
