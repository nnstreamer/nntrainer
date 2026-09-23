// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   compute_ops.h
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ComputeOps abstract interface for backend-agnostic dispatch
 *
 * Each Context (CPU/GPU/NPU) provides a concrete ComputeOps subclass.
 * Tensor operations call through this interface, enabling runtime
 * dispatch to the correct backend (ARM NEON, x86 AVX, OpenCL, CUDA,
 * QNN/HMX, ...) without #ifdef and — crucially — letting backend
 * subclasses carry their own state (cl_command_queue, npu_session,
 * kernel cache, ...) as member variables. That is the difference
 * between this and a function-pointer table: virtual dispatch lets
 * the impl reach back into per-backend resources without leaking a
 * `this` pointer through every call.
 *
 * Default method bodies throw std::runtime_error("not implemented").
 * Concrete subclasses override every op they want to support. For
 * accelerator-only ops (GPU batch/accel variants), pair the op with
 * a supports_*() predicate so callers can pick a CPU path on backends
 * that don't have an accelerated impl.
 */

#ifndef __COMPUTE_OPS_H__
#define __COMPUTE_OPS_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef ENABLE_FP16
#include <tensor_dim.h>
#endif

namespace nntrainer {

/**
 * @class ComputeOps
 * @brief Abstract dispatch interface for tensor compute kernels.
 */
class ComputeOps {
public:
  virtual ~ComputeOps() = default;

  // ===========================================================================
  // FP32 BLAS
  // ===========================================================================
  virtual void sgemm_fp32(const unsigned int TStorageOrder, bool TransA,
                          bool TransB, const unsigned int M,
                          const unsigned int N, const unsigned int K,
                          const float alpha, const float *A,
                          const unsigned int lda, const float *B,
                          const unsigned int ldb, const float beta, float *C,
                          const unsigned int ldc);

  virtual void sgemv_fp32(const unsigned int TStorageOrder, bool TransA,
                          const unsigned int M, const unsigned int N,
                          const float alpha, const float *A,
                          const unsigned int lda, const float *X,
                          const unsigned int incX, const float beta, float *Y,
                          const unsigned int incY);

  virtual float sdot_fp32(const unsigned int N, const float *X,
                          const unsigned int incX, const float *Y,
                          const unsigned int incY);

  virtual void saxpy_fp32(const unsigned int N, const float alpha,
                          const float *X, const unsigned int incX, float *Y,
                          const unsigned int incY);

  virtual void scopy_fp32(const unsigned int N, const float *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY);

  virtual void sscal_fp32(const unsigned int N, const float alpha, float *X,
                          const unsigned int incX);

  virtual float snrm2_fp32(const unsigned int N, const float *X,
                           const unsigned int incX);

  virtual unsigned int isamax_fp32(const unsigned int N, const float *X,
                                   const unsigned int incX);

  // ===========================================================================
  // FP32 Element-wise
  // ===========================================================================
  virtual void ele_mul_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_add_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_sub_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_div_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);

  // ===========================================================================
  // FP32 Activation / Special
  // ===========================================================================
  virtual void swiglu_fp32(const unsigned int N, float *X, float *Y, float *Z);
  virtual void swiglu_alpha_fp32(const unsigned int N, float *X, float *Y,
                                 float *Z, float alpha);
  virtual void tanh_gelu_fp32(const unsigned int N, const float *X, float *Y);
  virtual void gelu_v2_fp32(const unsigned int N, const float *X, float *Y);
  virtual void tanh_gelu_v2_fp32(const unsigned int N, const float *X,
                                 float *Y);
  virtual void tanh_gelu_mul_fp32(const unsigned int N, float *X, float *Y,
                                  float *Z);
  virtual void tanh_gelu_v2_mul_fp32(const unsigned int N, float *X, float *Y,
                                     float *Z);
  virtual float max_val_fp32(const unsigned int N, float *X);
  virtual void softmax_fp32(const unsigned int N, float *X, float *Y);
  virtual bool is_valid_fp32(const unsigned int N, const float *X);

  // ===========================================================================
  // FP32 Matrix ops
  // ===========================================================================
  virtual void transpose_matrix_fp32(const unsigned int M, const unsigned int N,
                                     const float *src, unsigned int ld_src,
                                     float *dst, unsigned int ld_dst);

  // ===========================================================================
  // FP32 Data conversion / Copy
  // ===========================================================================
  virtual void scopy_u8(const unsigned int N, const uint8_t *X,
                        const unsigned int incX, uint8_t *Y,
                        const unsigned int incY);
  virtual void scopy_s8(const unsigned int N, const int8_t *X,
                        const unsigned int incX, int8_t *Y,
                        const unsigned int incY);
  virtual void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                     const unsigned int incX, float *Y,
                                     const unsigned int incY);
  virtual void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y);
  virtual void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y);
  virtual void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y);
  virtual void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y);
  virtual void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y);
  virtual void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y);
  virtual void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y);

  // ===========================================================================
  // Quantized GEMM (GGUF format)
  // ===========================================================================
  virtual void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);
  virtual void gemm_q4_K_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);
  virtual void gemm_q6_K_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);

  // ===========================================================================
  // Quantized weight packing / quantization
  // ===========================================================================
  virtual void unpack_q4_0(const void *in_q4_0x, void *out_q4_0,
                           size_t data_size, const unsigned int M,
                           const unsigned int N);
  virtual void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                                         uint16_t *qs_out, int N, int K);
  virtual size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                               int64_t n_per_row, const float *quant_weights);
  virtual void dequantize_row_q4_0(const void *x, float *y, int64_t k);
  virtual void repack_q4_0(void *dst, void *src, size_t data_size,
                           const unsigned int M, const unsigned int N);

  // ===========================================================================
  // Clamp
  // ===========================================================================
  virtual void clamp_fp32(const float *input, float *output, size_t length,
                          float lower_bound, float upper_bound);

  // ===========================================================================
  // Data conversion (int8 → FP32)
  // ===========================================================================
  virtual void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                                    const unsigned int incX, float *Y,
                                    const unsigned int incY);
  virtual void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                                    const unsigned int incX, float *Y,
                                    const unsigned int incY);

  // ===========================================================================
  // Accelerator-only (GPU/NPU) ops — query supports_* before calling.
  // CPU subclasses leave both the impl (default-throw) and predicate
  // (default false) untouched. Accelerator subclasses override both.
  // ===========================================================================
  virtual bool supports_gemm_q4_0_batch_fp32() const { return false; }
  virtual void gemm_q4_0_batch_fp32(std::vector<void *> matAdata,
                                    float *matBdata,
                                    std::vector<float *> matCdata,
                                    unsigned int M, std::vector<unsigned int> N,
                                    unsigned int K);

  virtual bool supports_gemm_q4_0_accel_fp32() const { return false; }
  virtual void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata,
                                    float *matCdata, unsigned int M,
                                    unsigned int N, unsigned int K);

  // The M > 1 gate FloatTensor::dot() applies around both calls above
  // exists because, for CPU/GPU, a single-row GEMM is already fast
  // without going through the batch/accel path -- that reasoning does
  // not hold for HTP, where M == 1 (decode) is the FastRPC-amortization
  // shape these kernels exist for (see docs/htp_attention/
  // 40_moe_ffn_htp_task.md section2.2, 41_moe_ffn_e2e_and_perf_task.md
  // sectionB3). This predicate lets a backend opt in at M == 1 without
  // touching CPU/GPU's existing M > 1 behavior, which is intentional and
  // must not change.
  virtual bool accelerates_q4_0_at_m1() const { return false; }

  // QS4CX weights reach an accelerator without the Q4_0 detour: a model
  // quantized straight from FP32 into QS4CX carries the same int4 values
  // HexKL's registry wants, so the conversion at the seam is a bit
  // rearrangement plus a colsum rather than a second quantization (see
  // htp_qs4cx_from_packed). matAdata / matAscale are QS4CX_Tensor's
  // getData()/getScale() regions.
  virtual bool supports_gemm_qs4cx_accel_fp32() const { return false; }
  virtual void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale,
                                     float *matBdata, float *matCdata,
                                     unsigned int M, unsigned int N,
                                     unsigned int K);

  // Same grouping as gemm_q4_0_batch_fp32 (several weights sharing one
  // activation, e.g. decode's top-K experts' gate_up projections), for
  // QS4CX weights. Without this override, FloatTensor::dot's vector
  // overload had no fast path for QS4CX at all and fell through to a
  // per-weight loop of individual dotQs4cx() calls -- one FastRPC call per
  // expert instead of one per call, same shape of miss as the Q4_0 batch
  // override being written but the caller-side grouping shipping first
  // (htp_compute_ops.cpp's gemm_q4_0_batch_fp32 comment).
  virtual bool supports_gemm_qs4cx_batch_fp32() const { return false; }
  virtual void
  gemm_qs4cx_batch_fp32(std::vector<void *> matAdata,
                        std::vector<float *> matAscale, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

  // Fused MoE expert FFN (doc 43 §[L2]): ONE accelerator call computes
  // gate_up @ act -> SwiGLU -> down @, the SwiGLU intermediate never
  // leaving the accelerator. matAdata/matAscale/N carry exactly TWO QS4CX
  // weights, [gate_up (K x 2I), down (I x N_out)] -- N[0] == 2 * down.K is
  // the SwiGLU contract the impl must check. Callers gate on M > 1
  // themselves (decode's single token cannot amortize the fused call's
  // 64-row pad tax); without an override the caller keeps its two-dot +
  // host-activation fallback.
  virtual bool supports_gemm_qs4cx_fused_swiglu_fp32() const { return false; }
  virtual void gemm_qs4cx_fused_swiglu_fp32(std::vector<void *> matAdata,
                                            std::vector<float *> matAscale,
                                            float *matBdata, float *matCdata,
                                            unsigned int M,
                                            std::vector<unsigned int> N,
                                            unsigned int K);

  // A whole MoE FFN layer -- routing, every expert, and the scatter-add --
  // in one accelerator call (doc 46). Where gemm_qs4cx_fused_swiglu_fp32
  // above is one expert and the caller loops, this takes the routing table
  // and does the looping on the accelerator, so the token gather, the
  // routing multiply and the scatter-add stop crossing the boundary too.
  //
  // gate_up[e] is K x 2*inter and down[e] is inter x N_out. row_index holds
  // the token row each (expert, slot) works on, grouped by expert in expert
  // order; row_count says how many belong to each and must sum to
  // row_index's size; row_weight is the routing weight per entry. out is
  // M x N_out, zeroed by the implementation and accumulated into, because
  // experts share token rows.
  //
  // By const reference, unlike the vector-by-value overloads above: a
  // prefill layer's row_index is one entry per token per expert -- 1776 for
  // this model -- and copying that per layer is the kind of cost this call
  // exists to remove.
  //
  // w_bits says how the nibbles are stored: 0 = plain QS4CX (the backend
  // converts), 4 = QS4CX_WH (already in the HMX weight-tile layout),
  // 2 = QS2CX_WH (half that, expanded on the DSP -- doc 54)
  // (htp_wh_layout.h) with a per-output-channel column sum after the scales,
  // so the implementation registers them as they are instead of converting
  // and baking. It is a flag rather than a colsum pointer because the sums
  // sit immediately after the scales and the callee already knows N.
  virtual bool supports_gemm_qs4cx_moe_layer_fp32() const { return false; }
  virtual void
  gemm_qs4cx_moe_layer_fp32(const std::vector<void *> &gate_up_data,
                            const std::vector<float *> &gate_up_scale,
                            const std::vector<void *> &down_data,
                            const std::vector<float *> &down_scale,
                            const std::vector<unsigned int> &row_index,
                            const std::vector<unsigned int> &row_count,
                            const std::vector<float> &row_weight,
                            const float *act, float *out, unsigned int M,
                            unsigned int K, unsigned int inter,
                            unsigned int N_out, unsigned int w_bits);

  // Registers one K x N expert weight with the accelerator ahead of its
  // first use, so a model's load pays that cost rather than its first
  // prefill: for the 1408 weights of LFM2-8B-A1B it is 747 ms, 31% of the
  // measured prefill (doc 46 section 50.2 P6). The same call at forward time
  // then finds the weight already registered. A backend with nothing to
  // register returns false and the caller moves on.
  virtual bool register_qs4cx_weight(void *data, const float *scale,
                                     unsigned int K, unsigned int N,
                                     unsigned int w_bits) {
    (void)data;
    (void)scale;
    (void)K;
    (void)N;
    (void)w_bits;
    return false;
  }

  // The Q4_0 twin, for a fully_connected layer under engine=htp: its
  // weight is otherwise converted and registered by the first dot() that
  // reaches gemm_q4_0_accel_fp32, which is inside the first prefill (doc
  // 50). Called from Transformer::repack_weight with the data pointer that
  // dot() will pass, so the forward-time lookup is a cache hit.
  virtual bool register_q4_0_weight(void *data, unsigned int K,
                                    unsigned int N) {
    (void)data;
    (void)K;
    (void)N;
    return false;
  }

  // The dense SwiGLU FFN as ONE accelerator call (doc 51): up and gate
  // [K x I] and down [I x N], all Q4_0x4 as loaded; act [M x K] f32 ->
  // out [M x N] f32 = (silu(act . gate) * (act . up)) . down. The HTP
  // answers it with its MoE layer kernel over column chunks of I, each
  // chunk a whole "expert" whose down output is summed into out.
  // register_q4_0_dense_ffn is the load-time twin, like the two above.
  virtual bool supports_gemm_q4_0_dense_ffn_fp32() const { return false; }
  virtual void gemm_q4_0_dense_ffn_fp32(void *up, void *gate, void *down,
                                        const float *act, float *out,
                                        unsigned int M, unsigned int K,
                                        unsigned int I, unsigned int N) {
    (void)up;
    (void)gate;
    (void)down;
    (void)act;
    (void)out;
    (void)M;
    (void)K;
    (void)I;
    (void)N;
    throw std::runtime_error(
      "ComputeOps::gemm_q4_0_dense_ffn_fp32 not implemented by this backend");
  }
  virtual bool register_q4_0_dense_ffn(void *up, void *gate, void *down,
                                       unsigned int K, unsigned int I,
                                       unsigned int N) {
    (void)up;
    (void)gate;
    (void)down;
    (void)K;
    (void)I;
    (void)N;
    return false;
  }

  // An LFM2 conv block as ONE accelerator call (doc 51 section 2):
  // in_proj [K x 3C] and out_proj [C x N] Q4_0x4 as loaded, conv_w [3 x C]
  // f32 (w0 for row t, w1 for t-1, w2 for t-2); act [M x K] f32 ->
  // out [M x N] f32 = ((b * conv1d(a * c)) . out_proj) with a | b | c the
  // column thirds of act . in_proj, and state [2 x C] f32 <- the conv
  // input's rows M-2 and M-1 (zero where M is shorter), the state the
  // CPU decode path continues from. register_q4_0_conv_block is the
  // load-time twin.
  virtual bool supports_gemm_q4_0_conv_block_fp32() const { return false; }
  virtual void gemm_q4_0_conv_block_fp32(void *in_proj, const float *conv_w,
                                         void *out_proj, const float *act,
                                         float *out, float *state,
                                         unsigned int M, unsigned int K,
                                         unsigned int C, unsigned int N) {
    (void)in_proj;
    (void)conv_w;
    (void)out_proj;
    (void)act;
    (void)out;
    (void)state;
    (void)M;
    (void)K;
    (void)C;
    (void)N;
    throw std::runtime_error(
      "ComputeOps::gemm_q4_0_conv_block_fp32 not implemented by this backend");
  }
  virtual bool register_q4_0_conv_block(void *in_proj, void *out_proj,
                                        unsigned int K, unsigned int C,
                                        unsigned int N) {
    (void)in_proj;
    (void)out_proj;
    (void)K;
    (void)C;
    (void)N;
    return false;
  }

  virtual bool supports_gemv_int4_batch_fp32() const { return false; }
  virtual void gemv_int4_batch_fp32(std::vector<void *> weights,
                                    std::vector<uint16_t *> scales,
                                    float *input, std::vector<float *> outputs,
                                    unsigned int K,
                                    std::vector<unsigned int> Ns,
                                    unsigned int group_size);

  virtual bool supports_gemm_int4_batch_fp32() const { return false; }
  virtual void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                                    std::vector<uint16_t *> scales,
                                    std::vector<float *> matCdata,
                                    unsigned int M,
                                    std::vector<unsigned int> Ns,
                                    unsigned int K, unsigned int group_size);

  virtual bool supports_gemv_int4_accel_fp32() const { return false; }
  virtual void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                                    float *output, unsigned int K,
                                    unsigned int N, unsigned int group_size);

  virtual bool supports_sgemm_int4_accel_fp32() const { return false; }
  virtual void sgemm_int4_accel_fp32(float *input, char *weight,
                                     uint16_t *scale, float *output,
                                     unsigned int M, unsigned int N,
                                     unsigned int K, unsigned int group_size);

#ifdef ENABLE_FP16
  // ===========================================================================
  // FP16 BLAS
  // ===========================================================================
  virtual void sgemm_fp16(const unsigned int TStorageOrder, bool TransA,
                          bool TransB, const unsigned int M,
                          const unsigned int N, const unsigned int K,
                          const float alpha, const _FP16 *A,
                          const unsigned int lda, const _FP16 *B,
                          const unsigned int ldb, const float beta, _FP16 *C,
                          const unsigned int ldc);
  virtual void sgemv_fp16(const unsigned int TStorageOrder, bool TransA,
                          const unsigned int M, const unsigned int N,
                          const float alpha, const _FP16 *A,
                          const unsigned int lda, const _FP16 *X,
                          const unsigned int incX, const float beta, _FP16 *Y,
                          const unsigned int incY);
  virtual _FP16 sdot_fp16(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, const _FP16 *Y,
                          const unsigned int incY);
  virtual void saxpy_fp16(const unsigned int N, const float alpha,
                          const _FP16 *X, const unsigned int incX, _FP16 *Y,
                          const unsigned int incY);
  virtual void scopy_fp16(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY);
  virtual void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);
  virtual void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                                  const unsigned int incX, float *Y,
                                  const unsigned int incY);
  virtual void sscal_fp16(const unsigned int N, const float alpha, _FP16 *X,
                          const unsigned int incX);
  virtual _FP16 snrm2_fp16(const unsigned int N, const _FP16 *X,
                           const unsigned int incX);
  virtual unsigned int isamax_fp16(const unsigned int N, const _FP16 *X,
                                   const unsigned int incX);

  // ===========================================================================
  // FP16 Element-wise
  // ===========================================================================
  virtual void ele_mul_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_add_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_sub_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_div_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);

  // ===========================================================================
  // FP16 Activation / Special
  // ===========================================================================
  virtual void swiglu_fp16(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z);
  virtual _FP16 max_val_fp16(const unsigned int N, _FP16 *X);
  virtual void softmax_fp16(const unsigned int N, _FP16 *X, _FP16 *Y);
  virtual bool is_valid_fp16(const unsigned int N, const _FP16 *X);
  virtual void inv_sqrt_inplace_fp16(const unsigned int N, _FP16 *X);

  // ===========================================================================
  // FP16 Matrix ops
  // ===========================================================================
  virtual void transpose_matrix_fp16(const unsigned int M, const unsigned int N,
                                     const _FP16 *src, unsigned int ld_src,
                                     _FP16 *dst, unsigned int ld_dst);

  // ===========================================================================
  // FP16 Data conversion
  // ===========================================================================
  virtual void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                     const unsigned int incX, _FP16 *Y,
                                     const unsigned int incY);
  virtual void scopy_int8_to_float16_u(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, _FP16 *Y,
                                       const unsigned int incY);
  virtual void scopy_int8_to_float16_s(const unsigned int N, const int8_t *X,
                                       const unsigned int incX, _FP16 *Y,
                                       const unsigned int incY);

  // ===========================================================================
  // Mixed precision BLAS
  // ===========================================================================
  virtual void shgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const _FP16 *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc);
  virtual void shgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const _FP16 *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);
  virtual void hsgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const _FP16 *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc);
  virtual void hsgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const _FP16 *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);

  // ===========================================================================
  // Quantized GEMM (FP16 variants)
  // ===========================================================================
  virtual void gemm_q4_0_fp16(const unsigned int M, const unsigned int N,
                              const unsigned int K, const _FP16 *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, _FP16 *C,
                              const unsigned int ldc);
  virtual void gemm_q6_K_fp16(const unsigned int M, const unsigned int N,
                              const unsigned int K, const _FP16 *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, _FP16 *C,
                              const unsigned int ldc);

  // ===========================================================================
  // Rotary embedding
  // ===========================================================================
  virtual void compute_rotary_embedding_value(unsigned int dim,
                                              unsigned int half_,
                                              unsigned int w, _FP16 *in,
                                              _FP16 *out, float *cos_,
                                              float *sin_);
#endif // ENABLE_FP16

protected:
  /**
   * @brief Helper used by default impls to throw a uniform "not
   *        implemented" runtime_error tagged with the op name.
   */
  [[noreturn]] static void throwNotImplemented(const char *op);
};

/**
 * @brief Global compute ops pointer.
 *
 * Set once during init_backend(). When a Context-specific ops table is
 * available (via ContextData), that takes precedence.
 */
extern ComputeOps *g_compute_ops;

/**
 * @brief Ensure the global compute ops is initialized.
 */
void ensureComputeOps();

/**
 * @brief Get the active compute ops with lazy initialization.
 */
inline ComputeOps *getComputeOps() {
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_expect(g_compute_ops == nullptr, 0))
#else
  if (g_compute_ops == nullptr)
#endif
    ensureComputeOps();
  return g_compute_ops;
}

/**
 * @brief Initialize the CPU compute backend.
 *
 * Sets up architecture-specific resources (e.g., GGML, OpenBLAS threads)
 * and assigns g_compute_ops to the matching concrete ComputeOps
 * subclass for the current CPU architecture.
 */
void init_backend();

/**
 * @brief Backend-specific compute ops getters.
 *
 * `get_cpu_ops()` returns a process-wide singleton of the unified
 * `CpuComputeOps` subclass. The same singleton works for ARM / x86 /
 * fallback because each arch's compute_backend.cpp provides its own
 * specialised body for `nntrainer::sgemm` etc.; the wrapper class is
 * arch-agnostic and only needs to be defined once.
 */
ComputeOps *get_cpu_ops();
#ifdef ENABLE_OPENCL
/** @brief OpenCL accelerator ComputeOps singleton. Defined when
 *  enable-opencl is on, in cl_operations/cl_compute_ops.cpp. */
ComputeOps *get_cl_ops();
#endif
#ifdef ENABLE_HEXKL
/** @brief HTP (Hexagon/HMX) accelerator ComputeOps singleton. Defined
 *  when enable-htp is on, in htp_backend/htp_compute_ops.cpp. */
ComputeOps *get_htp_ops();
#endif

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __COMPUTE_OPS_H__ */
