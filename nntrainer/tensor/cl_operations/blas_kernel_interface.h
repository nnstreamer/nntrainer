// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.h
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNEL_INTERFACE_H__
#define __BLAS_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
Tensor dotCl(Tensor const &input, Tensor const &m, bool trans = false,
             bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotCl(Tensor const &input, Tensor const &m, Tensor &result,
           bool trans = false, bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans = false, bool trans_m = false);

/**
 * @brief Multiply value element by element immediately
 * @param[in] input Tensor
 * @param[in] value multiplier
 * @param[in] RunLayerContext reference
 */
void multiplyCl(Tensor &input, float const &value);

/**
 * @brief Process data and dimensions for add operation
 * @param[in] input Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 */
void add_i_cl(Tensor const &input, Tensor &result);

/**
 * @brief Process data and dimensions for transpose operation
 * @param[in] direction string
 * @param[in] input Tensor
 * @param[in] result Tensor
 */
void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result);

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
