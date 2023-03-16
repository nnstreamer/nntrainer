// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 seongwoo <mhs4670go@naver.com>
 *
 * @file   common.h
 * @date   18 May 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author seongwoo <mhs4670go@naver.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is common interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_COMMON_H__
#define __ML_TRAIN_COMMON_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <nntrainer-api-common.h>

/**
 * @brief Defining Macros for Exporting Dynamic Library in Windows.
 * @note Macro is meaningful only if platform is windows.
 */
#ifdef _WIN32
#define ML_API __declspec(dllexport)
#else
#define ML_API
#endif


namespace ml {
namespace train {

/**
 * @brief Defines Export Method to be called with
 *
 */
enum class ExportMethods {
  METHOD_STRINGVECTOR = 0, /**< export to a string vector */
  METHOD_TFLITE = 1,       /**< export to tflite */
  METHOD_FLATBUFFER = 2,   /**< export to flatbuffer */
  METHOD_UNDEFINED = 999,  /**< undefined */
};

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_COMMON_H__
