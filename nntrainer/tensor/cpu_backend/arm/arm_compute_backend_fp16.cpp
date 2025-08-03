// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file arm_compute_backend_fp16.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for arm
 *
 */
#include <arm_compute_backend.h>
#include <assert.h>
#include <fallback_internal.h>
#include <ggml_interface.h>
#include <neon_impl.h>
#include <nntrainer_error.h>

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

void sscal(const unsigned int N, const float alpha, _FP16 *X,
           const unsigned int incX) {
  assert(incX > 0);

  if (incX == 1) {
    nntrainer::neon::hscal(N, X, alpha);
  } else {
    __fallback_sscal(N, alpha, X, incX);
  }
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const unsigned int incX) {
  assert(incX > 0);
  _FP16 sum;
  if (incX == 1) {
    sum = nntrainer::neon::hnrm2(N, X);
  } else {
    sum = __fallback_snrm2(N, X, incX);
  }
  return sum;
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::hcopy(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp32_to_fp16(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp16_to_fp32(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int4_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
  }
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  _FP16 ret = 0;
  assert(incX > 0 && incY > 0);
  if (incX == 1 && incY == 1) {
    ret = nntrainer::neon::hdot(N, X, Y);
  } else {
    __fallback_sdot(N, X, incX, Y, incY);
  }
  return ret;
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const unsigned int incX, _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::haxpy(N, alpha, X, Y);
  } else {
    __fallback_saxpy(N, alpha, X, incX, Y, incY);
  }
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  if (TStorageOrder) {
    __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                     ldb, beta, C, ldc);
  } else {
    nntrainer::neon::custom_hgemm(A, B, C, M, N, K, alpha, beta, TransA,
                                  TransB);
  }
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const unsigned int incX,
           const float beta, _FP16 *Y, const unsigned int incY) {
  if (TStorageOrder) {
    __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                     Y, incY);
  } else {
    if (TransA) {
      nntrainer::neon::hgemv_transpose(A, X, Y, M, N, alpha, beta);
    } else {
      nntrainer::neon::hgemv(A, X, Y, M, N, alpha, beta);
    }
  }
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

unsigned int isamax(const unsigned int N, const _FP16 *X,
                    const unsigned int incX) {
  unsigned int max_idx = 0;
  if (incX == 1 && N >= 8) {
    max_idx = nntrainer::neon::isamax(N, X);
  } else {
    max_idx = __fallback_isamax(N, X, incX);
  }
  return max_idx;
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  nntrainer::neon::inv_sqrt_inplace(N, X);
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const __fp16 *src, unsigned int ld_src, __fp16 *dst,
                      unsigned int ld_dst) {
  nntrainer::neon::transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const __fp16 *input) {
  return nntrainer::neon::is_valid(N, input);
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
  nntrainer::neon::compute_rotary_embedding_value(dim, half_, w, in, out, cos_,
                                                  sin_);
}

void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  nntrainer::neon::swiglu(N, X, Y, Z);
}

_FP16 max_val(const unsigned int N, _FP16 *X) {
  return nntrainer::neon::max_val(N, X);
}

void softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
  nntrainer::neon::softmax(N, X, Y);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim) {
  neon::compute_fp16vcache_fp32_transposed(
    row_num, in, reinterpret_cast<const _FP16 *>(vcache), output,
    num_cache_head, gqa_size, head_dim);
}

template <>
void compute_kcaches(const float *A, const uint16_t *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  neon::compute_kcaches<_FP16>(A, reinterpret_cast<const _FP16 *>(B), output,
                               num_rows, N, chunk_size, group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_,
                                 only_convert_to_fp16);
}

void softmax_row_inplace(_FP16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads) {
  nntrainer::neon::softmax_row_inplace(qk_out, start_row, end_row, num_heads);
}

void softmax_row(_FP16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads) {
  nntrainer::neon::softmax_row(qk_out, start_row, end_row, num_heads);
}

void compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                   const _FP16 *vcache, _FP16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim) {
  neon::compute_fp16vcache_transposed(row_num, in, vcache, output,
                                      num_cache_head, gqa_size, head_dim);
}

void compute_kcaches(const _FP16 *A, const _FP16 *B, _FP16 *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  nntrainer::neon::compute_kcaches(A, B, output, num_rows, N, chunk_size,
                                   group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, _FP16 *inout,
                              const _FP16 *cos_, const _FP16 *sin_) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, cos_, sin_);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim) {
  neon::compute_fp16vcache_fp32_transposed(
    row_num, in, reinterpret_cast<const _FP16 *>(vcache), output,
    num_cache_head, gqa_size, head_dim);
}

template <>
void compute_kcaches(const float *A, const uint16_t *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  neon::compute_kcaches<_FP16>(A, reinterpret_cast<const _FP16 *>(B), output,
                               num_rows, N, chunk_size, group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_,
                                 only_convert_to_fp16);
}

void softmax_row_inplace(_FP16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads) {
  nntrainer::neon::softmax_row_inplace(qk_out, start_row, end_row, num_heads);
}

void softmax_row(_FP16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads) {
  nntrainer::neon::softmax_row(qk_out, start_row, end_row, num_heads);
}

void compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                   const _FP16 *vcache, _FP16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim) {
  neon::compute_fp16vcache_transposed(row_num, in, vcache, output,
                                      num_cache_head, gqa_size, head_dim);
}

void compute_kcaches(const _FP16 *A, const _FP16 *B, _FP16 *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  nntrainer::neon::compute_kcaches(A, B, output, num_rows, N, chunk_size,
                                   group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, _FP16 *inout,
                              const _FP16 *cos_, const _FP16 *sin_) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, cos_, sin_);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim) {
  neon::compute_fp16vcache_fp32_transposed(
    row_num, in, reinterpret_cast<const _FP16 *>(vcache), output,
    num_cache_head, gqa_size, head_dim);
}

template <>
void compute_kcaches(const float *A, const uint16_t *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  neon::compute_kcaches<_FP16>(A, reinterpret_cast<const _FP16 *>(B), output,
                               num_rows, N, chunk_size, group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_,
                                 only_convert_to_fp16);
}

void softmax_row_inplace(_FP16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads) {
  nntrainer::neon::softmax_row_inplace(qk_out, start_row, end_row, num_heads);
}

void softmax_row(_FP16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads) {
  nntrainer::neon::softmax_row(qk_out, start_row, end_row, num_heads);
}

void compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                   const _FP16 *vcache, _FP16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim) {
  neon::compute_fp16vcache_transposed(row_num, in, vcache, output,
                                      num_cache_head, gqa_size, head_dim);
}

void compute_kcaches(const _FP16 *A, const _FP16 *B, _FP16 *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  nntrainer::neon::compute_kcaches(A, B, output, num_rows, N, chunk_size,
                                   group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, _FP16 *inout,
                              const _FP16 *cos_, const _FP16 *sin_) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, cos_, sin_);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim) {
  neon::compute_fp16vcache_fp32_transposed(
    row_num, in, reinterpret_cast<const _FP16 *>(vcache), output,
    num_cache_head, gqa_size, head_dim);
}

template <>
void compute_kcaches(const float *A, const uint16_t *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  neon::compute_kcaches<_FP16>(A, reinterpret_cast<const _FP16 *>(B), output,
                               num_rows, N, chunk_size, group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_,
                                 only_convert_to_fp16);
}

void softmax_row_inplace(_FP16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads) {
  nntrainer::neon::softmax_row_inplace(qk_out, start_row, end_row, num_heads);
}

void softmax_row(_FP16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads) {
  nntrainer::neon::softmax_row(qk_out, start_row, end_row, num_heads);
}

void compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                   const _FP16 *vcache, _FP16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim) {
  neon::compute_fp16vcache_transposed(row_num, in, vcache, output,
                                      num_cache_head, gqa_size, head_dim);
}

void compute_kcaches(const _FP16 *A, const _FP16 *B, _FP16 *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
  nntrainer::neon::compute_kcaches(A, B, output, num_rows, N, chunk_size,
                                   group_size, tile_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, _FP16 *inout,
                              const _FP16 *cos_, const _FP16 *sin_) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, cos_, sin_);
}

template <>
size_t quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
#ifdef ENABLE_GGML
  return __nntr_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
#else
  return __fallback_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
#endif
}

template <> void dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k) {
#ifdef ENABLE_GGML
  __nntr_dequantize_row_q8_0(x_raw, y, k);
#else
  __fallback_dequantize_row_q8_0(x_raw, y, k);
#endif
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
#ifdef ENABLE_GGML
  return __ggml_q4_0_4x8_q8_0_GEMM<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
#else
  return __fallback_gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
#endif
}

template <> void quantize_row_q8_K(const _FP16 *src, void *dst, int64_t k) {
#ifdef ENABLE_GGML
  __ggml_quantize_row_q8_K(src, dst, k);
#else
  __fallback_quantize_row_q8_K(src, dst, k);
#endif
}

template <> void dequantize_row_q8_K(const void *x, _FP16 *y, int64_t k) {
#ifdef ENABLE_GGML
  __ggml_dequantize_row_q8_K(x, y, k);
#else
  __fallback_dequantize_row_q8_K(x, y, k);
#endif
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
#ifdef ENABLE_GGML
  return __ggml_gemm_q6_K<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
#else
  return __fallback_gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
#endif
}
} /* namespace nntrainer */
