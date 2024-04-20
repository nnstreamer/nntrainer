#include <assert.h>
#include <cmath>
#include <custom_fallback.h>
#include <fallback_internal.h>

namespace nntrainer {
void scopy(const unsigned int N, const uint8_t *X, const int incX, uint8_t *Y,
           const int intY) {
  __fallback_scopy(N, X, incX, Y, intY);
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int intY) {
  __fallback_scopy_int4_to_float32(N, X, incX, Y, intY);
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int intY) {
  __fallback_scopy_int8_to_float32(N, X, incX, Y, intY);
}

void sine(const unsigned int N, float *X, float *Y, float alpha) {
  __fallback_sine(N, X, Y, alpha);
}

void cosine(const unsigned int N, float *X, float *Y, float alpha) {
  __fallback_cosine(N, X, Y, alpha);
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  __fallback_inv_sqrt_inplace(N, X);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
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
#ifdef ENABLE_FP16
void sscal(const unsigned int N, const float alpha, _FP16 *X, const int incX) {
  __fallback_sscal(N, alpha, X, incX);
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const int incX) {
  __fallback_snrm2(N, X, incX);
}

void scopy(const unsigned int N, const _FP16 *X, const int incX, _FP16 *Y,
           const int incY) {
  __fallback_scopy(N, X, incX, incY);
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY) {
  __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY) {
  __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  __fallback_sdot(N, X, incX, Y, incY);
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const int incX, _FP16 *Y, const int incY) {
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_mul(N, X, Y, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_add(N, X, Y, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_sub(N, X, Y, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_div(N, X, Y, alpha, beta, i_stride, o_stride);
}

unsigned int isamax(const unsigned int N, const _FP16 *X, const int incX) {
  __fallback_isamax(N, X, incX)
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  __fallback_inv_sqrt_inplace(N, X);
}
#endif
} // namespace nntrainer
