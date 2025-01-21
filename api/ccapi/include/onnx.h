// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   onnx.h
 * @date   12 February 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#ifndef __ML_TRAIN_ONNX_H__
#define __ML_TRAIN_ONNX_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <onnx_interpreter.h>

namespace ml {
namespace train {

/**
 * @brief load model from onnx file
 *
 * @param path path of the onnx file to be loaded
 * @return std::unique_ptr<ml::train::Model>
 */
std::unique_ptr<ml::train::Model> loadONNX(const std::string &path);

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_ONNX_H__
