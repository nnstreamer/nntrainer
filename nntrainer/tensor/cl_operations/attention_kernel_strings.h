// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernel_strings.h
 * @date	8 October 2024
 * @brief	All attention OpenCL kernel strings
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ATTENTION_KERNEL_STRINGS_H__
#define __ATTENTION_KERNEL_STRINGS_H__

#include <string>

namespace nntrainer {

const std::string &getRotaryEmbClKernel();

#ifdef ENABLE_FP16

const std::string &getRotaryEmbClKernelFP16();

#endif

} // namespace nntrainer
#endif /* __ATTENTION_KERNEL_INTERFACE_H__ */
