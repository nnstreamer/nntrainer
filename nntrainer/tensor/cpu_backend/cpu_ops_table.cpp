// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cpu_ops_table.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unified CPU backend ComputeOps subclass.
 *
 * Single concrete ComputeOps subclass for ALL CPU targets (ARM /
 * x86 / fallback). The nntrainer::sgemm etc. functions are arch-
 * specialized — each arch_compute_backend.cpp defines its own body
 * — so a single forwarding wrapper is enough; build-time arch
 * dispatch picks the right symbol at link time.
 */

#include <cpu_ops_table.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <acti_func.h>
#include <tensor.h>

namespace nntrainer {

class CpuComputeOps : public ComputeOps {
public:
  // FP32 BLAS
  void sgemm_fp32(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const float *A,
                  unsigned int lda, const float *B, unsigned int ldb, float b,
                  float *C, unsigned int ldc) override {
    nntrainer::sgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp32(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const float *A, unsigned int lda, const float *X,
                  unsigned int iX, float b, float *Y,
                  unsigned int iY) override {
    nntrainer::sgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  float sdot_fp32(unsigned int N, const float *X, unsigned int iX,
                  const float *Y, unsigned int iY) override {
    return nntrainer::sdot(N, X, iX, Y, iY);
  }
  void saxpy_fp32(unsigned int N, float a, const float *X, unsigned int iX,
                  float *Y, unsigned int iY) override {
    nntrainer::saxpy(N, a, X, iX, Y, iY);
  }
  void scopy_fp32(unsigned int N, const float *X, unsigned int iX, float *Y,
                  unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void sscal_fp32(unsigned int N, float a, float *X, unsigned int iX) override {
    nntrainer::sscal(N, a, X, iX);
  }
  float snrm2_fp32(unsigned int N, const float *X, unsigned int iX) override {
    return nntrainer::snrm2(N, X, iX);
  }
  unsigned int isamax_fp32(unsigned int N, const float *X,
                           unsigned int iX) override {
    return nntrainer::isamax(N, X, iX);
  }

  // FP32 Element-wise
  void ele_mul_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_mul(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_add(N, X, Y, Z, a, b, is, os);
  }
  void ele_sub_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_sub(N, X, Y, Z, a, b, is, os);
  }
  void ele_div_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_div(N, X, Y, Z, a, b, is, os);
  }

  // FP32 Activation / Special
  void swiglu_fp32(unsigned int N, float *X, float *Y, float *Z) override {
    nntrainer::swiglu(N, X, Y, Z);
  }
  void swiglu_alpha_fp32(unsigned int N, float *X, float *Y, float *Z,
                         float alpha) override {
    nntrainer::swiglu(N, X, Y, Z, alpha);
  }
  void tanh_gelu_fp32(unsigned int N, const float *X, float *Y) override {
    nntrainer::tanh_gelu(N, X, Y);
  }
  void gelu_v2_fp32(unsigned int N, const float *X, float *Y) override {
    nntrainer::gelu_v2(N, X, Y);
  }
  void tanh_gelu_v2_fp32(unsigned int N, const float *X, float *Y) override {
    nntrainer::tanh_gelu_v2(N, X, Y);
  }
  void tanh_gelu_mul_fp32(unsigned int N, float *X, float *Y,
                          float *Z) override {
    nntrainer::tanh_gelu_mul(N, X, Y, Z);
  }
  void tanh_gelu_v2_mul_fp32(unsigned int N, float *X, float *Y,
                             float *Z) override {
    nntrainer::tanh_gelu_v2_mul(N, X, Y, Z);
  }
  float max_val_fp32(unsigned int N, float *X) override {
    return nntrainer::max_val(N, X);
  }
  void softmax_fp32(unsigned int N, float *X, float *Y) override {
    nntrainer::softmax(N, X, Y);
  }
  bool is_valid_fp32(unsigned int N, const float *X) override {
    return nntrainer::is_valid(N, X);
  }

  // FP32 Matrix
  void transpose_matrix_fp32(unsigned int M, unsigned int N, const float *s,
                             unsigned int lds, float *d,
                             unsigned int ldd) override {
    nntrainer::transpose_matrix(M, N, s, lds, d, ldd);
  }

  // FP32 Convolution
  void depthwise_conv2d_fp32(const float *input, const float *kernel,
                             float *output, unsigned int batch,
                             unsigned int channels, unsigned int in_h,
                             unsigned int in_w, unsigned int out_h,
                             unsigned int out_w, unsigned int kh,
                             unsigned int kw, unsigned int stride_h,
                             unsigned int stride_w, unsigned int pad_top,
                             unsigned int pad_left, unsigned int dilation_h,
                             unsigned int dilation_w) override {
    nntrainer::depthwise_conv2d_fp32(
      input, kernel, output, batch, channels, in_h, in_w, out_h, out_w, kh, kw,
      stride_h, stride_w, pad_top, pad_left, dilation_h, dilation_w);
  }

  // FP32 Data conversion / Copy
  void scopy_u8(unsigned int N, const uint8_t *X, unsigned int iX, uint8_t *Y,
                unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_s8(unsigned int N, const int8_t *X, unsigned int iX, int8_t *Y,
                unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_int4_to_float32(unsigned int N, const uint8_t *X, unsigned int iX,
                             float *Y, unsigned int iY) override {
    nntrainer::scopy_int4_to_float32(N, X, iX, Y, iY);
  }
  void copy_s16_fp32(unsigned int N, const int16_t *X, float *Y) override {
    nntrainer::copy_s16_fp32(N, X, Y);
  }
  void copy_u16_fp32(unsigned int N, const uint16_t *X, float *Y) override {
    nntrainer::copy_u16_fp32(N, X, Y);
  }
  void copy_fp32_u32(unsigned int N, const float *X, uint32_t *Y) override {
    nntrainer::copy_fp32_u32(N, X, Y);
  }
  void copy_fp32_u16(unsigned int N, const float *X, uint16_t *Y) override {
    nntrainer::copy_fp32_u16(N, X, Y);
  }
  void copy_fp32_u8(unsigned int N, const float *X, uint8_t *Y) override {
    nntrainer::copy_fp32_u8(N, X, Y);
  }
  void copy_fp32_s16(unsigned int N, const float *X, int16_t *Y) override {
    nntrainer::copy_fp32_s16(N, X, Y);
  }
  void copy_fp32_s8(unsigned int N, const float *X, int8_t *Y) override {
    nntrainer::copy_fp32_s8(N, X, Y);
  }

  // Quantized GEMM
  void gemm_q4_0_fp32(unsigned int M, unsigned int N, unsigned int K,
                      const float *A, unsigned int lda, const void *B,
                      unsigned int ldb, float *C, unsigned int ldc) override {
    nntrainer::gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q4_K_fp32(unsigned int M, unsigned int N, unsigned int K,
                      const float *A, unsigned int lda, const void *B,
                      unsigned int ldb, float *C, unsigned int ldc) override {
    nntrainer::gemm_q4_K(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q6_K_fp32(unsigned int M, unsigned int N, unsigned int K,
                      const float *A, unsigned int lda, const void *B,
                      unsigned int ldb, float *C, unsigned int ldc) override {
    nntrainer::gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
  }
  // im2col-fused q4_0 conv GEMM — only the ARM i8mm backend implements the
  // fused gather+quantize path; elsewhere callers fall back via supports_*().
  bool supports_gemm_q4_0_indirect_conv_fp32() const override {
    return NNTR_HAS_Q4_0_INDIRECT_CONV;
  }
  void gemm_q4_0_indirect_conv_fp32(unsigned int M, unsigned int N,
                                    unsigned int K, const float *in,
                                    const ConvGatherParams &geom, const void *B,
                                    unsigned int ldb, float *C,
                                    unsigned int ldc) override {
    nntrainer::gemm_q4_0_indirect_conv(M, N, K, in, geom, B, ldb, C, ldc);
  }

  // Quantization / Utility
  void unpack_q4_0(const void *in, void *out, size_t ds, unsigned int M,
                   unsigned int N) override {
    nntrainer::unpack_q4_0(in, out, ds, M, N);
  }
  void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                                 uint16_t *qs_out, int N, int K) override {
    nntrainer::unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
  }
  size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                       int64_t n_per_row, const float *qw) override {
    return nntrainer::quantize_q4_0(src, dst, nrow, n_per_row, qw);
  }
  void dequantize_row_q4_0(const void *x, float *y, int64_t k) override {
    nntrainer::dequantize_row_q4_0(x, y, k);
  }
  void repack_q4_0(void *dst, void *src, size_t ds, unsigned int M,
                   unsigned int N) override {
    nntrainer::repack_q4_0(dst, src, ds, M, N);
  }

  void clamp_fp32(const float *in, float *out, size_t len, float lb,
                  float ub) override {
    nntrainer::clamp(in, out, len, lb, ub);
  }

  void scopy_int8_to_fp32_u(unsigned int N, const uint8_t *X, unsigned int iX,
                            float *Y, unsigned int iY) override {
    nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
  }
  void scopy_int8_to_fp32_s(unsigned int N, const int8_t *X, unsigned int iX,
                            float *Y, unsigned int iY) override {
    nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
  }

  // Accelerator-only ops: CPU backends do not implement these — base
  // class defaults (throw + supports_*() = false) are correct.

#ifdef ENABLE_FP16
  void sgemm_fp16(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const _FP16 *A,
                  unsigned int lda, const _FP16 *B, unsigned int ldb, float b,
                  _FP16 *C, unsigned int ldc) override {
    nntrainer::sgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp16(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const _FP16 *A, unsigned int lda, const _FP16 *X,
                  unsigned int iX, float b, _FP16 *Y,
                  unsigned int iY) override {
    nntrainer::sgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  _FP16 sdot_fp16(unsigned int N, const _FP16 *X, unsigned int iX,
                  const _FP16 *Y, unsigned int iY) override {
    return nntrainer::sdot(N, X, iX, Y, iY);
  }
  void saxpy_fp16(unsigned int N, float a, const _FP16 *X, unsigned int iX,
                  _FP16 *Y, unsigned int iY) override {
    nntrainer::saxpy(N, a, X, iX, Y, iY);
  }
  void scopy_fp16(unsigned int N, const _FP16 *X, unsigned int iX, _FP16 *Y,
                  unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_fp32_to_fp16(unsigned int N, const float *X, unsigned int iX,
                          _FP16 *Y, unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void scopy_fp16_to_fp32(unsigned int N, const _FP16 *X, unsigned int iX,
                          float *Y, unsigned int iY) override {
    nntrainer::scopy(N, X, iX, Y, iY);
  }
  void sscal_fp16(unsigned int N, float a, _FP16 *X, unsigned int iX) override {
    nntrainer::sscal(N, a, X, iX);
  }
  _FP16 snrm2_fp16(unsigned int N, const _FP16 *X, unsigned int iX) override {
    return nntrainer::snrm2(N, X, iX);
  }
  unsigned int isamax_fp16(unsigned int N, const _FP16 *X,
                           unsigned int iX) override {
    return nntrainer::isamax(N, X, iX);
  }

  void ele_mul_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_mul(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_add(N, X, Y, Z, a, b, is, os);
  }
  void ele_sub_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_sub(N, X, Y, Z, a, b, is, os);
  }
  void ele_div_fp16(unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    nntrainer::ele_div(N, X, Y, Z, a, b, is, os);
  }

  void swiglu_fp16(unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) override {
    nntrainer::swiglu(N, X, Y, Z);
  }
  _FP16 max_val_fp16(unsigned int N, _FP16 *X) override {
    return nntrainer::max_val(N, X);
  }
  void softmax_fp16(unsigned int N, _FP16 *X, _FP16 *Y) override {
    nntrainer::softmax(N, X, Y);
  }
  bool is_valid_fp16(unsigned int N, const _FP16 *X) override {
    return nntrainer::is_valid(N, X);
  }
  void inv_sqrt_inplace_fp16(unsigned int N, _FP16 *X) override {
    nntrainer::inv_sqrt_inplace(N, X);
  }

  void transpose_matrix_fp16(unsigned int M, unsigned int N, const _FP16 *s,
                             unsigned int lds, _FP16 *d,
                             unsigned int ldd) override {
    nntrainer::transpose_matrix(M, N, s, lds, d, ldd);
  }

  void scopy_int4_to_float16(unsigned int N, const uint8_t *X, unsigned int iX,
                             _FP16 *Y, unsigned int iY) override {
    nntrainer::scopy_int4_to_float16(N, X, iX, Y, iY);
  }
  void scopy_int8_to_float16_u(unsigned int N, const uint8_t *X,
                               unsigned int iX, _FP16 *Y,
                               unsigned int iY) override {
    nntrainer::scopy_int8_to_float16(N, X, iX, Y, iY);
  }
  void scopy_int8_to_float16_s(unsigned int N, const int8_t *X, unsigned int iX,
                               _FP16 *Y, unsigned int iY) override {
    nntrainer::scopy_int8_to_float16(N, X, iX, Y, iY);
  }

  void shgemm(unsigned int o, bool tA, bool tB, unsigned int M, unsigned int N,
              unsigned int K, float a, const float *A, unsigned int lda,
              const _FP16 *B, unsigned int ldb, float b, float *C,
              unsigned int ldc) override {
    nntrainer::shgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void shgemv(unsigned int o, bool tA, unsigned int M, unsigned int N, float a,
              const float *A, unsigned int lda, const _FP16 *X, unsigned int iX,
              float b, float *Y, unsigned int iY) override {
    nntrainer::shgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  void hsgemm(unsigned int o, bool tA, bool tB, unsigned int M, unsigned int N,
              unsigned int K, float a, const _FP16 *A, unsigned int lda,
              const float *B, unsigned int ldb, float b, float *C,
              unsigned int ldc) override {
    nntrainer::hsgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void hsgemv(unsigned int o, bool tA, unsigned int M, unsigned int N, float a,
              const _FP16 *A, unsigned int lda, const float *X, unsigned int iX,
              float b, float *Y, unsigned int iY) override {
    nntrainer::hsgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }

  void gemm_q4_0_fp16(unsigned int M, unsigned int N, unsigned int K,
                      const _FP16 *A, unsigned int lda, const void *B,
                      unsigned int ldb, _FP16 *C, unsigned int ldc) override {
    nntrainer::gemm_q4_0<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
  }
  // im2col-fused q4_0 conv GEMM, FP16 activation — FP16 mirror of the FP32
  // indirect conv above. Same availability gate (NNTR_HAS_Q4_0_INDIRECT_CONV);
  // the FP16 backend op is provided under the same macro (see
  // arm_compute_backend gemm_q4_0_indirect_conv_fp16 + the
  // __ggml..._indirect_GEMM_fp16 specialization).
  bool supports_gemm_q4_0_indirect_conv_fp16() const override {
    return NNTR_HAS_Q4_0_INDIRECT_CONV;
  }
  void gemm_q4_0_indirect_conv_fp16(unsigned int M, unsigned int N,
                                    unsigned int K, const _FP16 *in,
                                    const ConvGatherParams &geom, const void *B,
                                    unsigned int ldb, _FP16 *C,
                                    unsigned int ldc) override {
    nntrainer::gemm_q4_0_indirect_conv_fp16(M, N, K, in, geom, B, ldb, C, ldc);
  }
  void gemm_q6_K_fp16(unsigned int M, unsigned int N, unsigned int K,
                      const _FP16 *A, unsigned int lda, const void *B,
                      unsigned int ldb, _FP16 *C, unsigned int ldc) override {
    nntrainer::gemm_q6_K<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
  }

  void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                      unsigned int w, _FP16 *in, _FP16 *out,
                                      float *cos_, float *sin_) override {
    nntrainer::compute_rotary_embedding_value(dim, half_, w, in, out, cos_,
                                              sin_);
  }

  // FP16 Convolution
  void depthwise_conv2d_fp16(const _FP16 *input, const float *kernel,
                             _FP16 *output, unsigned int batch,
                             unsigned int channels, unsigned int in_h,
                             unsigned int in_w, unsigned int out_h,
                             unsigned int out_w, unsigned int kh,
                             unsigned int kw, unsigned int stride_h,
                             unsigned int stride_w, unsigned int pad_top,
                             unsigned int pad_left, unsigned int dilation_h,
                             unsigned int dilation_w) override {
    nntrainer::depthwise_conv2d_fp16(
      input, kernel, output, batch, channels, in_h, in_w, out_h, out_w, kh, kw,
      stride_h, stride_w, pad_top, pad_left, dilation_h, dilation_w);
  }
#endif // ENABLE_FP16
};

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}

namespace {

/**
 * @todo These scalar helpers are the reference form of the activation, and the
 * baseline every other backend table is checked against. A vectorized form
 * (NEON/AVX, riding the link-time arch dispatch that nntrainer::sgemm already
 * uses) is intended but not written yet.
 */

/** gelu, tanh approximation (gelu_pytorch_tanh) */
inline float gelu_tanh_f(float x) {
  constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
/** silu, in the numerically stable x / (1 + exp(-x)) form */
inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
/**
 * logistic sigmoid. The OpenCL sigmoid_glu / sigmoid_add kernels and the CUDA
 * ELTWISE_SRC ones spell it the same way (1/(1+exp(-x))) so that the three
 * backends agree token for token rather than only to within a rounding.
 */
inline float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/**
 * @brief Run a binary element-wise op over the row window, dispatching on the
 *        activation dtype once instead of per element. FP16 operands are
 *        computed in float and rounded back, so the two dtypes agree.
 * @todo This is a scalar reference implementation, and also the cross-backend
 * correctness baseline. NEON/AVX specializations can ride the existing
 * link-time arch dispatch (the nntrainer::sgemm pattern), and row-level
 * threading applies to prefill windows; neither is done yet.
 */
template <typename Op>
void elementwise2_rows(const Tensor &in1, const Tensor &in2, Tensor &out,
                       unsigned int active_rows, unsigned int row_offset,
                       const char *op_name, Op op) {
  const unsigned int width = in1.width();
  const size_t elem_off = (size_t)row_offset * width;
  const size_t n = (size_t)active_rows * width;

  // These are the reference implementations every backend table is checked
  // against, and they index in2/out with in1's dtype and stride, so a
  // disagreeing operand would be silently reinterpreted rather than rejected.
  if (in2.getDim() != in1.getDim() || out.getDim() != in1.getDim())
    throw std::invalid_argument(std::string("CpuComputeOps::") + op_name +
                                ": operands must share shape and dtype");

  if (n == 0)
    return;

  switch (in1.getDataType()) {
  case ml::train::TensorDim::DataType::FP32: {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = op(a[i], b[i]);
    break;
  }
#ifdef ENABLE_FP16
  case ml::train::TensorDim::DataType::FP16: {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(op((float)a[i], (float)b[i]));
    break;
  }
#endif
  default:
    throw std::invalid_argument(std::string("CpuComputeOps::") + op_name +
                                ": unsupported data type");
  }
}

/**
 * @brief (x - mean) * rsqrt(var + eps) * gamma + beta over `rows` rows of
 *        `width` elements. Mean and variance accumulate in FP32 regardless of
 *        the operand dtype.
 * @param T activation dtype
 * @param G weight dtype of gamma/beta, which need not equal T -- the core
 *          LayerNormalizationLayer requests them at the weight dtype.
 * @todo Scalar reference implementation (also the cross-backend correctness
 * baseline). An arch-specialized version can follow the existing
 * rms_norm_wrt_width_*_intrinsic pattern (x86/avx2_impl.h, arm/neon_impl.h) --
 * LayerNorm adds mean subtraction and gamma/beta but the row-wise reduction
 * structure is identical; row-level threading applies for prefill windows.
 * Not yet implemented.
 */
template <typename T, typename G>
void layernorm_rows(const T *x, const G *g, const G *b, T *y, unsigned int rows,
                    unsigned int width, float eps) {
  for (unsigned int r = 0; r < rows; ++r) {
    const T *xr = x + (size_t)r * width;
    T *yr = y + (size_t)r * width;

    float mean = 0.0f;
    for (unsigned int k = 0; k < width; ++k)
      mean += (float)xr[k];
    mean /= (float)width;

    float ssd = 0.0f;
    for (unsigned int k = 0; k < width; ++k) {
      const float d = (float)xr[k] - mean;
      ssd += d * d;
    }
    const float inv = 1.0f / std::sqrt(ssd / (float)width + eps);

    for (unsigned int k = 0; k < width; ++k)
      yr[k] = (T)((((float)xr[k] - mean) * inv) * (float)g[k] + (float)b[k]);
  }
}

/**
 * @brief GELU computed in float. The constants match __fallback_gelu_v2 /
 *        __fallback_tanh_gelu so the FP16 path agrees with the FP32 one.
 * @param tanh_mode false = erf-exact GELU, true = tanh approximation
 * @todo This is a scalar reference implementation, and also the cross-backend
 * correctness baseline. A NEON/AVX specialization can ride the existing
 * link-time arch dispatch (the nntrainer::sgemm pattern) the way the FP32 path
 * already reaches nntrainer::gelu_v2; it is not done yet.
 */
template <typename T>
void gelu_elems(const T *x, T *y, size_t n, bool tanh_mode) {
  for (size_t i = 0; i < n; ++i) {
    const float v = (float)x[i];
    const float r =
      tanh_mode
        ? 0.5f * v *
            (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)))
        : 0.5f * v * (1.0f + std::erf(v * 0.7071067811f));
    y[i] = (T)r;
  }
}

} // namespace

void CpuComputeOps::geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                          unsigned int active_rows, unsigned int row_offset) {
  elementwise2_rows(in1, in2, out, active_rows, row_offset, "geglu",
                    [](float a, float b) { return gelu_tanh_f(a) * b; });
}

void CpuComputeOps::swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset) {
  elementwise2_rows(in1, in2, out, active_rows, row_offset, "swiglu",
                    [](float a, float b) { return silu_f(a) * b; });
}

void CpuComputeOps::sigmoid_glu(const Tensor &in1, const Tensor &in2,
                                Tensor &out, unsigned int active_rows,
                                unsigned int row_offset) {
  elementwise2_rows(in1, in2, out, active_rows, row_offset, "sigmoid_glu",
                    [](float a, float b) { return sigmoid_f(a) * b; });
}

void CpuComputeOps::sigmoid_add(const Tensor &in1, const Tensor &in2,
                                Tensor &out, unsigned int active_rows,
                                unsigned int row_offset) {
  elementwise2_rows(in1, in2, out, active_rows, row_offset, "sigmoid_add",
                    [](float a, float b) { return sigmoid_f(a) + b; });
}

void CpuComputeOps::layer_norm(const Tensor &in, Tensor &out,
                               const Tensor &gamma, const Tensor &beta,
                               float epsilon, unsigned int active_rows,
                               unsigned int row_offset) {
  using DT = ml::train::TensorDim::DataType;
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  const DT dt = in.getDataType();
  const DT gt = gamma.getDataType(); // gamma and beta always share one dtype

  if (active_rows == 0 || width == 0)
    return;

  if (dt == DT::FP32 && gt == DT::FP32) {
    layernorm_rows(in.getData<float>() + elem_off, gamma.getData<float>(),
                   beta.getData<float>(), out.getData<float>() + elem_off,
                   active_rows, width, epsilon);
#ifdef ENABLE_FP16
  } else if (dt == DT::FP16 && gt == DT::FP16) {
    layernorm_rows(in.getData<_FP16>() + elem_off, gamma.getData<_FP16>(),
                   beta.getData<_FP16>(), out.getData<_FP16>() + elem_off,
                   active_rows, width, epsilon);
  } else if (dt == DT::FP16 && gt == DT::FP32) {
    layernorm_rows(in.getData<_FP16>() + elem_off, gamma.getData<float>(),
                   beta.getData<float>(), out.getData<_FP16>() + elem_off,
                   active_rows, width, epsilon);
  } else if (dt == DT::FP32 && gt == DT::FP16) {
    layernorm_rows(in.getData<float>() + elem_off, gamma.getData<_FP16>(),
                   beta.getData<_FP16>(), out.getData<float>() + elem_off,
                   active_rows, width, epsilon);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::layer_norm: unsupported data type");
  }
}

/**
 * GELU and tanh-GELU get an explicit dtype switch instead of going through
 * ActiFunc, because ActiFunc::gelu/tanhGelu call nntrainer::gelu_v2/tanh_gelu
 * with t_in.getData<float>() unconditionally: on an FP16 activation tensor
 * that reinterprets half data as float and reads size() * 4 bytes out of a
 * size() * 2 byte buffer. The FP32 path below calls the very same backend
 * functions ActiFunc does, so FP32 output is unchanged.
 *
 * Every other mode is delegated to ActiFunc so there is ONE host
 * implementation of the activation family and every existing mode stays
 * value-identical. A partial row window is expressed as a shared-data view,
 * which is safe here because this is by definition the host path.
 */
void CpuComputeOps::activation(const Tensor &in, Tensor &out, int act_type,
                               unsigned int active_rows,
                               unsigned int row_offset) {
  using DT = ml::train::TensorDim::DataType;
  const auto at = static_cast<ActivationType>(act_type);
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  const size_t n = (size_t)active_rows * width;
  const DT dt = in.getDataType();

  if (n == 0)
    return;

  if (at == ActivationType::ACT_GELU || at == ActivationType::ACT_TANH_GELU) {
    const bool tanh_mode = (at == ActivationType::ACT_TANH_GELU);
    if (dt == DT::FP32) {
      const float *x = in.getData<float>() + elem_off;
      float *y = out.getData<float>() + elem_off;
      if (tanh_mode)
        nntrainer::tanh_gelu((unsigned int)n, x, y);
      else
        nntrainer::gelu_v2((unsigned int)n, x, y);
      return;
    }
#ifdef ENABLE_FP16
    if (dt == DT::FP16) {
      gelu_elems(in.getData<_FP16>() + elem_off,
                 out.getData<_FP16>() + elem_off, n, tanh_mode);
      return;
    }
#endif
    throw std::invalid_argument(
      "CpuComputeOps::activation: unsupported data type for gelu");
  }

  if (at == ActivationType::ACT_NONE) {
    if (in.getData<uint8_t>() == out.getData<uint8_t>())
      return;
    size_t esz;
    if (dt == DT::FP32)
      esz = sizeof(float);
    else if (dt == DT::FP16)
      esz = 2u;
    else
      throw std::invalid_argument(
        "CpuComputeOps::activation: unsupported data type");
    std::memcpy(out.getData<uint8_t>() + elem_off * esz,
                in.getData<uint8_t>() + elem_off * esz, n * esz);
    return;
  }

  ActiFunc f;
  if (dt == DT::FP16) {
#ifdef ENABLE_FP16
    f.setActiFunc<_FP16>(at);
#else
    throw std::invalid_argument(
      "CpuComputeOps::activation: fp16 needs enable-fp16");
#endif
  } else {
    f.setActiFunc<float>(at);
  }

  const TensorDim d = in.getDim();
  const unsigned int total_rows = d.batch() * d.channel() * d.height();
  const bool aliased = (in.getData<uint8_t>() == out.getData<uint8_t>());

  if (row_offset == 0 && active_rows == total_rows) {
    if (aliased && !f.supportInPlace()) {
      Tensor in_copy = out.clone();
      f.run_fn(in_copy, out);
    } else {
      f.run_fn(in, out);
    }
    return;
  }

  TensorDim wd = d;
  wd.batch(1);
  wd.channel(1);
  wd.height(active_rows);
  Tensor in_view = in.getSharedDataTensor(wd, elem_off, true);
  Tensor out_view = out.getSharedDataTensor(wd, elem_off, true);
  if (aliased && !f.supportInPlace()) {
    Tensor in_copy = in_view.clone();
    f.run_fn(in_copy, out_view);
  } else {
    f.run_fn(in_view, out_view);
  }
}

void CpuComputeOps::residual_op(Tensor &hidden, const Tensor &input,
                                bool accumulate) {
  if (accumulate)
    hidden.add_i(input);
  else
    hidden.copy(input);
}

void CpuComputeOps::fc(Tensor &input, Tensor &weight, Tensor &output) {
  input.dot(weight, output, false, false);
}

void CpuComputeOps::apply_activation(Tensor &out, int act_type) {
  if (static_cast<ActivationType>(act_type) == ActivationType::ACT_NONE)
    return;
  const TensorDim d = out.getDim();
  activation(out, out, act_type, d.batch() * d.channel() * d.height(),
             /*row_offset=*/0);
}

void CpuComputeOps::mean_rows(const Tensor &in, Tensor &out,
                              unsigned int active_rows,
                              unsigned int row_offset) {
  // Reduce through a [1, 1, active_rows, W] window and average over axis 2 --
  // exactly the shape and call a mean-pooling caller built by hand before this
  // op existed, so the reduction order (and therefore the result) is unchanged.
  const unsigned int W = in.width();
  Tensor rows =
    in.getSharedDataTensor({1, 1, active_rows, W}, (size_t)row_offset * W);
  out.copyData(rows.average(2));
}

void CpuComputeOps::l2_normalize_rows(const Tensor &in, Tensor &out,
                                      float epsilon) {
  // Same rationale as mean_rows: this is the copyData + normalization_i(3)
  // pair the callers ran before, so host numerics are unchanged.
  if (out.getData<uint8_t>() != in.getData<uint8_t>())
    out.copyData(in);
  out.normalization_i(3, 2.0f, epsilon);
}

} // namespace nntrainer
