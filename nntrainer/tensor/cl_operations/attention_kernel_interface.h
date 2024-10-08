// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	blas_kernel_interface.h
 * @date	28 August 2024
 * @brief	Interface for attention OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ATTENTION_KERNEL_INTERFACE_H__
#define __ATTENTION_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Rotary Embedding kernel
 * @param[in] in input tensor
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep maximum timestep
 */
void apply_rotary_emb_cl(Tensor &in, unsigned int dim, unsigned int from,
                         unsigned int max_timestep);

} // namespace nntrainer
#endif /* __ATTENTION_KERNEL_INTERFACE_H__ */
