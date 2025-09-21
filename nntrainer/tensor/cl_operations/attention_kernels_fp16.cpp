// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels_fp16.cpp
 * @date	28 August 2024
 * @brief	Common attention OpenCL fp16 kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "attention_kernels_templates.h"
#include <cl_kernels/cl_kernels.h>

#include <cassert>

namespace nntrainer {

#define CL_CHECK(expression)                                                   \
  do {                                                                         \
    if (!(expression)) {                                                       \
      ml_loge("Expression %s failed", #expression);                            \
    }                                                                          \
  } while (false)

void rotary_emb_cl(_FP16 *in, _FP16 *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestep, unsigned int in_size,
                   unsigned int out_size) {
  bool result = false;

  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &cl_buffer_manager = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_rotaryEmb_fp16_ptr =
    cl_context->registerClKernel(rotary_emb_fp16_kernel, "rotary_emb_cl_fp16");
  if (!kernel_rotaryEmb_fp16_ptr) {
    return;
  }

  rotary_emb_cl_internal<_FP16>(
    kernel_rotaryEmb_fp16_ptr, in, out, freqs_cos, freqs_sin, cos_, sin_, batch,
    channel, height, width, dim, from, max_timestep, in_size, out_size);
}

void attention_cl_fp16(void *q, void *k, void *s, void *v, void *o,
                       const uint32_t m, const uint32_t n, const uint32_t d_k,
                       const uint32_t d_v) {
  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &cl_buffer_manager = ClBufferManager::Global();
  ClContext::SharedPtrClKernel kernel_qkt =
    cl_context->registerClKernel(attn_f16_kernel, "attn_f16_q_mul_transpose_k");

  if (!kernel_qkt) {
    return;
  }

  ClContext::SharedPtrClKernel kernel_smr =
    cl_context->registerClKernel(attn_f16_kernel, "attn_f16_softmax_row");

  if (!kernel_smr) {
    return;
  }

  ClContext::SharedPtrClKernel kernel_sm_v =
    cl_context->registerClKernel(attn_f16_kernel, "attn_f16_sm_mul_v");

  if (!kernel_sm_v) {
    return;
  }

  const float inverse_sqrt_d_k = 1.0f / std::sqrt(static_cast<float>(d_k));

  CL_CHECK(kernel_qkt->SetKernelSVMArguments(0, q));
  CL_CHECK(kernel_qkt->SetKernelSVMArguments(1, k));
  CL_CHECK(kernel_qkt->SetKernelSVMArguments(2, s));
  CL_CHECK(kernel_qkt->SetKernelArguments(3, &n, sizeof(int)));
  CL_CHECK(kernel_qkt->SetKernelArguments(4, &d_k, sizeof(int)));
  CL_CHECK(kernel_qkt->SetKernelArguments(5, &inverse_sqrt_d_k, sizeof(float)));

  CL_CHECK(kernel_smr->SetKernelSVMArguments(0, s));
  CL_CHECK(kernel_smr->SetKernelArguments(1, &m, sizeof(int)));
  CL_CHECK(kernel_smr->SetKernelArguments(2, &n, sizeof(int)));

  CL_CHECK(kernel_sm_v->SetKernelSVMArguments(0, s));
  CL_CHECK(kernel_sm_v->SetKernelSVMArguments(1, v));
  CL_CHECK(kernel_sm_v->SetKernelSVMArguments(2, o));
  CL_CHECK(kernel_sm_v->SetKernelArguments(3, &n, sizeof(int)));
  CL_CHECK(kernel_sm_v->SetKernelArguments(4, &d_v, sizeof(int)));

  const int tile_size = 16;
  const int kernel_qkt_global_size[3] = {(int)m, (int)n};
  const int kernel_qkt_local_size[3] = {tile_size, tile_size, 1};

  const int kernel_smr_global_size[3] = {(int)m, (int)1};
  const int kernel_smr_local_size[3] = {tile_size, 1, 1};

  const int kernel_smv_global_size[3] = {(int)m, (int)d_v};
  const int kernel_smv_local_size[3] = {tile_size, tile_size, 1};

  cl_event qkt_event = NULL;
  cl_event smr_event = NULL;

  CL_CHECK(cl_context->command_queue_inst_.DispatchCommand(
    kernel_qkt, kernel_qkt_global_size, kernel_qkt_local_size, &qkt_event));
  CL_CHECK(cl_context->command_queue_inst_.DispatchCommand(
    kernel_smr, kernel_smr_global_size, kernel_smr_local_size, &smr_event,
    {qkt_event}));
  CL_CHECK(cl_context->command_queue_inst_.DispatchCommand(
    kernel_sm_v, kernel_smv_global_size, kernel_smv_local_size, nullptr,
    {smr_event}));

  clWaitForEvents(1, &smr_event);
  clReleaseEvent(qkt_event);
  clReleaseEvent(smr_event);
}
} // namespace nntrainer
