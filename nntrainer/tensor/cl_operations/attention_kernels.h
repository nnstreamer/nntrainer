// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels.h
 * @date	28 August 2024
 * @brief	Common attention OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ATTENTION_KERNELS_H__
#define __ATTENTION_KERNELS_H__

#include <layer_context.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <string>

namespace nntrainer {

/**
 * @brief declaring global kernel objects
 */
extern opencl::Kernel kernel_rotary_emb;

/**
 * @brief     Rotary Embedding process
 * @param[in] in __fp16 * input
 * @param[in] out __fp16 * output
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[in] cos_ vector of cos values
 * @param[in] sin_ vector of sin values
 * @param[in] batch size of batch
 * @param[in] channel channel of input
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep max timestep
 * @param[in] in_size size of input
 * @param[in] out_size size of output
 * @param[in] context RunLayerContext reference
 */
void rotary_emb_cl(float *in, float *out,
                   std::vector<std::vector<float>> freqs_cos,
                   std::vector<std::vector<float>> freqs_sin,
                   std::vector<float> cos_, std::vector<float> sin_,
                   unsigned int batch, unsigned int channel,
                   unsigned int height, unsigned int width, unsigned int dim,
                   unsigned int from, unsigned int max_timestamp,
                   unsigned int in_size, unsigned int out_size,
                   RunLayerContext &context);

#ifdef ENABLE_FP16
/**
 * @brief declaring global fp16 kernel objects
 */
extern opencl::Kernel kernel_rotary_emb_fp16;

/**
 * @brief     Rotary Embedding process
 * @param[in] in __fp16 * input
 * @param[in] out __fp16 * output
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[in] cos_ vector of cos values
 * @param[in] sin_ vector of sin values
 * @param[in] batch size of batch
 * @param[in] channel channel of input
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep max timestep
 * @param[in] in_size size of input
 * @param[in] out_size size of output
 * @param[in] context RunLayerContext reference
 */
void rotary_emb_cl(__fp16 *in, __fp16 *out,
                   std::vector<std::vector<float>> freqs_cos,
                   std::vector<std::vector<float>> freqs_sin,
                   std::vector<float> cos_, std::vector<float> sin_,
                   unsigned int batch, unsigned int channel,
                   unsigned int height, unsigned int width, unsigned int dim,
                   unsigned int from, unsigned int max_timestamp,
                   unsigned int in_size, unsigned int out_size,
                   RunLayerContext &context);

#endif

} // namespace nntrainer
#endif /* __ATTENTION_KERNELS_H__ */