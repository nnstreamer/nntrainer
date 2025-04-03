// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_strings.h
 * @date	18 Sep 2024
 * @brief	All blas OpenCL kernel strings
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNEL_STRINGS_H__
#define __BLAS_KERNEL_STRINGS_H__

#include <string>

namespace nntrainer {

const std::string &getSgemvClKernel();

const std::string &getSgemvClNoTransKernel();

const std::string &getDotClKernel();

const std::string &getSgemmClNoTransKernel();

const std::string &getSgemmClTransAKernel();

const std::string &getSgemmClTransBKernel();

const std::string &getSgemmClTransABKernel();

const std::string &getAdditionClKernel();

const std::string &getSscalClKernel();

const std::string &getTransposeClKernelAxis0();

const std::string &getTransposeClKernelAxis1();

const std::string &getTransposeClKernelAxis2();

const std::string &getSwiGluClKernel();

const std::string &getCopyClKernel();

const std::string &getConcatClAxis3Kernel();

const std::string &getConcatClAxis2Kernel();

const std::string &getConcatClAxis1Kernel();

const std::string &getRMSNormClKernel();

#ifdef ENABLE_FP16

const std::string &getHgemvClKernel();

const std::string &getHgemvClNoTransKernel();

const std::string &getDotClKernelFP16();

const std::string &getHgemmClNoTransKernel();

const std::string &getHgemmClTransAKernel();

const std::string &getHgemmClTransBKernel();

const std::string &getHgemmClTransABKernel();

const std::string &getAdditionClKernelFP16();

const std::string &getHscalClKernel();

const std::string &getTransposeClAxis0KernelFP16();

const std::string &getTransposeClAxis1KernelFP16();

const std::string &getTransposeClAxis2KernelFP16();

const std::string &getSwiGluClKernelFP16();

const std::string &getCopyClKernelFP16();

const std::string &getConcatClAxis3KernelFP16();

const std::string &getConcatClAxis2KernelFP16();

const std::string &getConcatClAxis1KernelFP16();

const std::string &getRMSNormClKernelFP16();

#endif
} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
