#ifndef __CUSTOM_FALLBACK_H__
#define __CUSTOM_FALLBACK_H__
#ifdef __cplusplus

#include <stdint.h>

namespace nntrainer{
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy(const unsigned int N, const uint8_t *X, const int incX,
                  uint8_t *Y, const int intY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                  const int incX, float *Y, const int intY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                                  const int incX, float *Y, const int intY);

/**
 * @brief     sine with neon: Y = sin(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void sine(const unsigned int N, float *X, float *Y, float alpha = 1.f);

/**
 * @brief     cosine with neon: Y = cos(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void cosine(const unsigned int N, float *X, float *Y, float alpha = 1.f);

/**
 * @brief inversed squared root transformation inplace : X = 1 / sqrt(X)
 *
 * @param N size of X
 * @param X float * for Vector X
 */
void inv_sqrt_inplace(const unsigned int N, float *X);
/**
 * @brief     elementwise vector multiplication : Z = X ⊙ alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_mul(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector addition : Z = X + alpha * Y + beta *
 * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_add(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);
/**
 * @brief     elementwise vector subtraction with neon : Z = X - alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) + beta
 * * Z
 * @note ZeroDivisionError is not guaranteed in this function
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);
#ifdef ENABLE_FP16
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] alpha float number
 */
void sscal(const unsigned int N, const float alpha, _FP16 *X,
                  const int incX);

/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
_FP16 snrm2(const unsigned int N, const _FP16 *X, const int incX);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy(const unsigned int N, const _FP16 *X, const int incX,
                  _FP16 *Y, const int incY);



/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                  const int incX, _FP16 *Y, const int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                                  const int incX, _FP16 *Y, const int incY);

/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  const _FP16 *Y, const unsigned int incY);

/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
                  const int incX, _FP16 *Y, const int incY);

/**
 * @brief     elementwise vector multiplication : Z = X ⊙ alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector addition : Z = X + alpha * Y + beta *
 * Z
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);
/**
 * @brief     elementwise vector subtraction with neon : Z = X - alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) + beta
 * * Z
 * @note ZeroDivisionError is not guaranteed in this function
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
unsigned int isamax(const unsigned int N, const _FP16 *X,
                           const int incX);

/**
 * @brief squared root transformation inplace : X = sqrt(X)
 *
 * @param N size of X
 * @param X __fp16 * for Vector X
 */
void inv_sqrt_inplace(const unsigned int N, _FP16 *X);
#endif
}

#endif
#endif
