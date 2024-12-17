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
 * @brief  fused process data and dimensions for OpenCL dot operation, addition
 * and RMS
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] bias Tensor
 * @param[in] disable_bias_value bool
 * @param[in] gamma Tensor
 * @param[in] epsilon float
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
Tensor fusedProcess(Tensor const &input, Tensor const &m, Tensor const &bias,
                    bool disable_bias_value, Tensor const &gamma,
                    const float epsilon, bool trans = false,
                    bool trans_m = false);

/**
 * @brief  fused process data and dimensions for OpenCL dot operation, addition
 * and RMS
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] bias Tensor
 * @param[in] disable_bias_value bool
 * @param[in] gamma Tensor
 * @param[in] epsilon float
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void fusedProcess(Tensor const &input, Tensor const &m, Tensor &result,
                  Tensor const &bias, bool disable_bias_value,
                  Tensor const &gamma, const float epsilon, bool trans = false,
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
 * @param[in] result Tensor
 * @param[in] input Tensor
 */
void add_i_cl(Tensor &result, Tensor const &input);

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
