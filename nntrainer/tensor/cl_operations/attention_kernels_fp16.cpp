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
#include <cl_kernels/rotary_emb_fp16.h>

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

void flash_attention_cl_fp16(
  // clang-format off
                             void *q, void *k, void *v, void *o,
                             const uint32_t q_width, const uint32_t q_height, const uint32_t q_channels, const uint32_t q_batches,
                             const uint32_t k_width, const uint32_t k_height, const uint32_t k_channels, const uint32_t k_batches,
                             const uint32_t v_width, const uint32_t v_height, const uint32_t v_channels, const uint32_t v_batches,
                             const uint32_t o_width, const uint32_t o_height, const uint32_t o_channels, const uint32_t o_batches
  // clang-format on
) {
  // clang-format off
  // NOTE : beloved GGML uses 'ne' array corrsponding to [W, H, C, N (what is N ???)] https://github.com/ggml-org/ggml/issues/500#issuecomment-1704322898
  // clang-format on

  // clang-format off
  auto *cl_context        = static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &cl_buffer_manager = ClBufferManager::Global();
  ClContext::SharedPtrClKernel kernel = cl_context->registerClKernel(getFlashAttentionClKernelFP16(), "flash_attn_f16");
  // clang-format on

  if (!kernel) {
    return;
  }

  // SET FROM M, N, K
  unsigned long q_ne0 = q_width;
  unsigned long q_ne1 = q_height;
  unsigned long q_ne2 = q_channels;

  unsigned long k_ne0 = k_width;
  unsigned long k_ne1 = k_height;
  unsigned long k_ne2 = k_channels;

  unsigned long v_ne0 = v_width;
  unsigned long v_ne1 = v_height;
  unsigned long v_ne2 = v_channels;

  unsigned long o_ne0 = o_width;
  unsigned long o_ne1 = o_height;
  unsigned long o_ne2 = o_channels;

  // TODO(mwlasiuk) : set ... ???
  float max_bias = 0;
  float logit_softcap = 0;
  float scale = 0;

  // DIRECT
  // clang-format off
  const int d_head_q = q_ne0 ;      // q->ne[0];
  const int n_q = q_ne1 ;           // q->ne[1];
  const int n_head = q_ne2 ;        // q->ne[2];
  const int n_batch = q_batches;    // q->ne[3];
  //
  const int n_kv = k_ne1 ;          // k->ne[1];
  const int n_head_kv = k_ne2 ;     // k->ne[2];
  //
  const int d_head_v = o_ne0 ;      // v->ne[0];
  // clang-format on

  // clang-format off
  int n_head_log2 = n_head > 0 ? 1u << (int)std::floor(std::log2((float)n_head)) : 0;
  // clang-format on

  // clang-format off
  const int n_head_log2_val = n_head > 0 ? 1u << (int)std::floor(std::log2((float)n_head)) : 0;
  const float n_head_log2_f = n_head_log2_val > 0 ? (float)n_head_log2_val : 1.0f;
  const float m0 = powf(2.0f, -(max_bias) / n_head_log2_f);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2_f);
  // clang-format on

  int is_causal = (n_q > 1 && n_q == n_kv);

  unsigned long q_nb0 = sizeof(_FP16);
  unsigned long q_nb1 = q_nb0 * (q_ne0 / 1);
  unsigned long q_nb2 = q_nb1 * q_ne1;
  unsigned long q_nb3 = q_nb2 * q_ne2;

  unsigned long k_nb0 = sizeof(_FP16);
  unsigned long k_nb1 = k_nb0 * (k_ne0 / 1);
  unsigned long k_nb2 = k_nb1 * k_ne1;
  unsigned long k_nb3 = k_nb2 * k_ne2;

  unsigned long v_nb0 = sizeof(_FP16);
  unsigned long v_nb1 = v_nb0 * (v_ne0 / 1);
  unsigned long v_nb2 = v_nb1 * v_ne1;
  unsigned long v_nb3 = v_nb2 * v_ne2;

  unsigned long o_nb0 = sizeof(_FP16);
  unsigned long o_nb1 = o_nb0 * (o_ne0 / 1);
  unsigned long o_nb2 = o_nb1 * o_ne1;
  unsigned long o_nb3 = o_nb2 * o_ne2;

  // clang-format off
  CL_CHECK(kernel->SetKernelSVMArguments(/*  [0]  const global void  * q_void        */ 0,  q));
  CL_CHECK(kernel->SetKernelSVMArguments(/*  [2]  const global void  * k_void        */ 1,  k));
  CL_CHECK(kernel->SetKernelSVMArguments(/*  [4]  const global void  * v_void        */ 2,  v));
  CL_CHECK(kernel->SetKernelSVMArguments(/*  [6]        global void  * o_void        */ 3,  o));
  CL_CHECK(kernel->SetKernelArguments(/*     [8]  const        float   scale         */ 4,  &scale, sizeof(float)));
  CL_CHECK(kernel->SetKernelArguments(/*     [9]  const        int     n_q           */ 5,  &n_q, sizeof(int)));
  CL_CHECK(kernel->SetKernelArguments(/*     [10] const        int     n_kv          */ 6, &n_kv, sizeof(int)));
  CL_CHECK(kernel->SetKernelArguments(/*     [11] const        int     is_causal     */ 7, &is_causal, sizeof(int)));
  CL_CHECK(kernel->SetKernelArguments(/*     [12] const        int     n_head        */ 8, &n_head, sizeof(int)));
  CL_CHECK(kernel->SetKernelArguments(/*     [13] const        ulong   q_nb1         */ 9, &q_nb1, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [14] const        ulong   q_nb2         */ 10, &q_nb2, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [15] const        ulong   q_nb3         */ 11, &q_nb3, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [16] const        ulong   k_nb1         */ 12, &k_nb1, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [17] const        ulong   k_nb2         */ 13, &k_nb2, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [18] const        ulong   k_nb3         */ 14, &k_nb3, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [19] const        ulong   v_nb1         */ 15, &v_nb1, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [20] const        ulong   v_nb2         */ 16, &v_nb2, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [21] const        ulong   v_nb3         */ 17, &v_nb3, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [22] const        ulong   o_nb1         */ 18, &o_nb1, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [23] const        ulong   o_nb2         */ 19, &o_nb2, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [24] const        ulong   o_nb3         */ 20, &o_nb3, sizeof(unsigned long)));
  CL_CHECK(kernel->SetKernelArguments(/*     [25] const        float   max_bias      */ 21, &max_bias, sizeof(float)));
  CL_CHECK(kernel->SetKernelArguments(/*     [26] const        float   m0            */ 22, &m0, sizeof(float)));
  CL_CHECK(kernel->SetKernelArguments(/*     [27] const        float   m1            */ 23, &m1, sizeof(float)));
  CL_CHECK(kernel->SetKernelArguments(/*     [28] const        int     n_head_log2   */ 24, &n_head_log2_val, sizeof(int)));
  CL_CHECK(kernel->SetKernelArguments(/*     [29] const        float   logit_softcap */ 25, &logit_softcap, sizeof(float)));
  CL_CHECK(kernel->SetKernelArguments(/*     [30] const        int     n_head_kv     */ 26, &n_head_kv, sizeof(int)));
  // clang-format on

  std::printf(" - q =                %p\n", q);
  std::printf(" - k =                %p\n", k);
  std::printf(" - v =                %p\n", v);
  std::printf(" - o =                %p\n", o);
  std::printf(" - scale =            %f\n", scale);
  std::printf(" - n_q =              %d\n", n_q);
  std::printf(" - n_kv =             %d\n", n_kv);
  std::printf(" - is_causal =        %d\n", is_causal);
  std::printf(" - n_head =           %d\n", n_head);
  std::printf(" - q_nb1 =            %lu\n", q_nb1);
  std::printf(" - q_nb2 =            %lu\n", q_nb2);
  std::printf(" - q_nb3 =            %lu\n", q_nb3);
  std::printf(" - k_nb1 =            %lu\n", k_nb1);
  std::printf(" - k_nb2 =            %lu\n", k_nb2);
  std::printf(" - k_nb3 =            %lu\n", k_nb3);
  std::printf(" - v_nb1 =            %lu\n", v_nb1);
  std::printf(" - v_nb2 =            %lu\n", v_nb2);
  std::printf(" - v_nb3 =            %lu\n", v_nb3);
  std::printf(" - o_nb1 =            %lu\n", o_nb1);
  std::printf(" - o_nb2 =            %lu\n", o_nb2);
  std::printf(" - o_nb3 =            %lu\n", o_nb3);
  std::printf(" - max_bias =         %f\n", max_bias);
  std::printf(" - m0 =               %f\n", m0);
  std::printf(" - m1 =               %f\n", m1);
  std::printf(" - n_head_log2_val =  %d\n", n_head_log2_val);
  std::printf(" - logit_softcap =    %f\n", logit_softcap);
  std::printf(" - n_head_kv =        %d\n", n_head_kv);

  if (n_q == 1) {
    // TODO(mwlasiuk) : q1
    //
    // const size_t wg_size = 64;
    // size_t local_work_size[] = {wg_size, 1};
    // size_t global_work_size[] = {wg_size, (size_t)(n_head * n_batch)};
    // backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size,
    //                                     local_work_size, dst);
  } else {
    // TODO(mwlasiuk) : !q1
    //
    const int block_m = 16; // HARDOCED
    const int wg_size = block_m;

    const int local_work_size[3] = {(int)wg_size, 1, 1};
    const int global_work_size[3] = {(int)((n_q + block_m - 1) / block_m) *
                                       wg_size,
                                     (int)(n_head * n_batch), 1};

    std::printf("Running L = [%d, %d, %d] G = [%d, %d, %d]\n",
                local_work_size[0], local_work_size[1], local_work_size[2],
                global_work_size[0], global_work_size[1], global_work_size[2]);

    CL_CHECK(cl_context->command_queue_inst_.DispatchCommand(
      kernel, global_work_size, local_work_size));
  }
}
} // namespace nntrainer
