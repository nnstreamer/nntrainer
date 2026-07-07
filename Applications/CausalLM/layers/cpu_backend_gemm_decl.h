// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cpu_backend_gemm_decl.h
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Forward declarations of libnntrainer GEMM kernels used by the flash
 *         attention path in mha_core.cpp.
 *
 * @note   libnntrainer.so is built with ENABLE_FP16=1 and exports these GEMM
 *         kernels; the CausalLM app may be built with ENABLE_FP16=0, in which
 *         case cpu_backend's headers hide them behind #ifdef. Re-declaring the
 *         ones we call here lets mha_core.cpp compile regardless of its
 *         ENABLE_FP16 flag; symbols resolve at link time. See
 *         FLASH_ATTENTION.md.
 *
 *         - shgemm:                  FP32 A × FP16 B -> FP32 C  (FP32 accum)
 *         - custom_hgemm:            FP16 A × FP16 B -> FP16 C  (FP16 result)
 *         - hgemm_f16xf16_f32_fmlal: FP16 A × FP16 B -> FP32 C  (QK, FMLAL)
 *         - hsgemm_fp16bits_avx2:    FP32 A × FP16-bits B -> FP32 C (AVX2+F16C)
 */
#ifndef __CPU_BACKEND_GEMM_DECL_H__
#define __CPU_BACKEND_GEMM_DECL_H__
#pragma once

#include <cstdint>

#if defined(__ARM_NEON)
#include <neon_mathfun.h> // exp_ps: 4-wide Cephes NEON exp
#endif

#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
namespace nntrainer {
void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const __fp16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc);
namespace neon {
void custom_hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                  uint32_t N, uint32_t K, float alpha, float beta, bool TransA,
                  bool TransB);
} // namespace neon
} // namespace nntrainer
void hgemm_f16xf16_f32_fmlal(const __fp16 *A, const __fp16 *B, float *C,
                             unsigned int M, unsigned int N, unsigned int K,
                             float alpha, unsigned int lda, unsigned int ldb,
                             unsigned int ldc);
#endif

#if defined(__x86_64__) || defined(__i386__)
namespace nntrainer {
namespace avx2 {
void hsgemm_fp16bits_avx2(unsigned int M, unsigned int N, unsigned int K,
                          float alpha, const float *A, unsigned int lda,
                          const uint16_t *B, unsigned int ldb, bool TransB,
                          float *C, unsigned int ldc);
} // namespace avx2
} // namespace nntrainer
#endif

#endif // __CPU_BACKEND_GEMM_DECL_H__
