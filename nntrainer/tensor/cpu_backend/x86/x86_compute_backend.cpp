// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   x86_compute_backend.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for x86
 *
 */

#include <assert.h>

#include <avx2_impl.h>
#ifdef USE_BLAS
#include <cblas_interface.h>
#endif
#include <fallback_internal.h>
#include <ggml_interface.h>
#include <nntrainer_error.h>
#include <x86_compute_backend.h>

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

void init_backend() { __ggml_init(); }

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int4_to_float32(N, X, incX, Y, incY);
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  __fallback_copy_s16(N, X, Y);
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  __fallback_copy_u16(N, X, Y);
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  __fallback_copy_s16_fp32(N, X, Y);
}

void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y) {
  nntrainer::avx2::copy_f16_f32(N, X, Y);
}

void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y) {
  __fallback_copy_fp32_u32(N, X, Y);
}

void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y) {
  nntrainer::avx2::copy_f32_f16(N, X, Y);
}

void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) {
  __fallback_copy_fp32_u8(N, X, Y);
}

void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y) {
  __fallback_copy_fp32_s16(N, X, Y);
}

void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) {
  __fallback_copy_fp32_s8(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint32_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint32_t *Y) {
  copy_fp32_u32(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint16_t *Y) {
  copy_fp32_u16(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint8_t *Y) {
  copy_fp32_u8(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, int16_t *Y) {
  copy_fp32_s16(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, int8_t *Y) {
  copy_fp32_s8(N, X, Y);
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_uint8_to_float32(N, X, incX, Y, incY);
}

void scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float32(N, X, incX, Y, incY);
}

template <>
void sine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  __fallback_sine(N, X, Y, alpha, beta);
}

template <>
void cosine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  __fallback_cosine(N, X, Y, alpha, beta);
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  __fallback_inv_sqrt_inplace(N, X);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  nntrainer::avx2::ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  nntrainer::avx2::ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  __cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
#endif
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const unsigned int incX,
           const float beta, float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                incY);
#else
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
#endif
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  return __cblas_sdot(N, X, incX, Y, incY);
#else
  return __fallback_sdot(N, X, incX, Y, incY);
#endif
}

void scopy(const unsigned int N, const uint8_t *X, const unsigned int incX,
           uint8_t *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const int8_t *X, const unsigned int incX,
           int8_t *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  /// @note cblas_scopy is evoking SIGSEGV for some reason. Use custom
  /// implementation instead.
  // __cblas_scopy(N, X, incX, Y, incY);
  nntrainer::avx2::custom_scopy(N, X, incX, Y, incY);
}

void sscal(const unsigned int N, const float alpha, float *X,
           const unsigned int incX) {
#ifdef USE_BLAS
  __cblas_sscal(N, alpha, X, incX);
#else
  __fallback_sscal(N, alpha, X, incX);
#endif
}

float snrm2(const unsigned int N, const float *X, const unsigned int incX) {
#ifdef USE_BLAS
  return __cblas_snrm2(N, X, incX);
#else
  return __fallback_snrm2(N, X, incX);
#endif
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {
#ifdef USE_BLAS
  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc);
#else
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
#endif
}

unsigned int isamax(const unsigned int N, const float *X,
                    const unsigned int incX) {
#ifdef USE_BLAS
  return __cblas_isamax(N, X, incX);
#else
  return __fallback_isamax(N, X, incX);
#endif
}
void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  nntrainer::avx2::transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const float *input) {
  return nntrainer::avx2::is_valid(N, input);
}

void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                               uint16_t *qs_out, int N, int K) {
  return nntrainer::avx2::unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from,
                                 float attention_scaling) {
  __fallback_calc_trigonometric_vals_dup(N_half, angle, cos_, sin_, from,
                                         attention_scaling);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  nntrainer::avx2::swiglu(N, X, Y, Z);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z, float alpha) {
  nntrainer::avx2::swiglu(N, X, Y, Z, alpha);
}

float max_val(const unsigned int N, float *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, float *X, float *Y) {
  __fallback_softmax(N, X, Y);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_0_8x8_q8_0_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_0(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  throw std::runtime_error("Error: NYI for gemm_q4_0 with vectored weights");
}

void gemm_q4_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_K_8x8_q8_K_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_K(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  return __ggml_q4_K_8x8_q8_K_GEMM(M, Ns, K, A, lda, Bs, ldbs, Cs, ldcs);
}

float dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                    const void *v_q8_K) {
  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_K);
}

float dot_q6_K_f32(const unsigned int K, const void *v_q6_K, const float *f) {
  return __ggml_vec_dot_q6_K_f32(K, v_q6_K, f);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
}

size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q4_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q6_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

void quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_row_q6_K(src, dst, k);
}

template <> void quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_row_q8_K(src, dst, k);
}

void dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q4_K(x_raw, y, k);
}

void dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q4_0(x_raw, y, k);
}

void dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q6_K(x, y, k);
}

template <> void dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q8_K(x, y, k);
}

void repack_q4_0(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                           const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_K(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_K_to_q4_K_8(W, repacked_W, data_size, M, N);
}

template <>
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  nntrainer::avx2::softmax_row_inplace<float>(qk_out, start_row, end_row,
                                              num_heads, sink);
}

template <>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  nntrainer::avx2::softmax_row<float>(qk_out, start_row, end_row, num_heads,
                                      sink);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim,
                                        size_t local_window_size) {
  nntrainer::avx2::compute_fp16vcache_fp32_transposed(
    row_num, in, vcache, output, num_cache_head, gqa_size, head_dim,
    local_window_size);
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size) {
  nntrainer::avx2::compute_kcaches<uint16_t>(in, kcache, output, num_rows,
                                             num_cache_head, head_dim, gqa_size,
                                             tile_size, local_window_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  nntrainer::avx2::compute_rotary_emb_value(width, dim, half_, inout, output,
                                            cos_, sin_, only_convert_to_fp16);
}

void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  nntrainer::avx2::rms_norm_wrt_width_fp32_intrinsic(X, Y, H, W, epsilon);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp16_intrinsic(X, Y, H, W, epsilon);
}

template <>
void clamp(const float *input, float *output, size_t length, float lower_bound,
           float upper_bound) {
  nntrainer::avx2::clamp(input, output, length, lower_bound, upper_bound);
}

void nntr_quant_qs4cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                          void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                          bool transB) {
  __fallback_nntr_quant_qs4cx_f32(n, k, rhs_native_mtx_f32,
                                  rhs_native_mtx_qs4cx, rhs_scales_f32, transB);
}

template <>
void nntr_gemm_qai8dxp_qsi4cxp(size_t m, size_t n, size_t k,
                               void *lhs_native_mtx_f32,
                               void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                               float *dst_mtx_f32, bool transB,
                               float lower_bound, float upper_bound) {
  __fallback_nntr_gemm_qai8dxp_qsi4cxp(
    m, n, k, lhs_native_mtx_f32, rhs_native_mtx_qs4cx, rhs_scales_f32,
    dst_mtx_f32, transB, lower_bound, upper_bound);
}
} /* namespace nntrainer */
