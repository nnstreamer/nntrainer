// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel.h
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a collection of all the KERNELs function for hgemm
 *
 */

/**
 * @brief hgemm_kernel_8x16 KERNEL function
 *
 * @param M Length of blocked M
 * @param N Length of blocked N
 * @param K Length of blocked K
 * @param sa Starting address of blocked A
 * @param sb Starting address of blocked B
 * @param sc Starting address of blocked C
 * @param ldc Leading dimension of original matrix C
 */
template <typename T>
void hgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, T *sc, unsigned int ldc);

/**
 * @brief hgemm_kernel_8x8 KERNEL function
 *
 * @param M Length of blocked M
 * @param N Length of blocked N
 * @param K Length of blocked K
 * @param sa Starting address of blocked A
 * @param sb Starting address of blocked B
 * @param sc Starting address of blocked C
 * @param ldc Leading dimension of original matrix C
 */
template <typename T>
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, T *sc, unsigned int ldc);

/**
 * @brief hgemm_kernel_4x8 KERNEL function
 *
 * @param M Length of blocked M
 * @param N Length of blocked N
 * @param K Length of blocked K
 * @param sa Starting address of blocked A
 * @param sb Starting address of blocked B
 * @param sc Starting address of blocked C
 * @param ldc Leading dimension of original matrix C
 */
template <typename T>
void hgemm_kernel_4x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, T *sc, unsigned int ldc);

/**
 * @brief hgemm_kernel_4x4 KERNEL function
 *
 * @param M Length of blocked M
 * @param N Length of blocked N
 * @param K Length of blocked K
 * @param sa Starting address of blocked A
 * @param sb Starting address of blocked B
 * @param sc Starting address of blocked C
 * @param ldc Leading dimension of original matrix C
 */
template <typename T>
void hgemm_kernel_4x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, T *sc, unsigned int ldc);

/**
 * @brief hgemm_kernel_1x8 KERNEL function
 *
 * @param M Length of blocked M
 * @param N Length of blocked N
 * @param K Length of blocked K
 * @param sa Starting address of blocked A
 * @param sb Starting address of blocked B
 * @param sc Starting address of blocked C
 * @param ldc Leading dimension of original matrix C
 */
template <typename T>
void hgemm_kernel_1x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, T *sc, unsigned int ldc);

/**
 * @brief hgemm_kernel_1x4 KERNEL function
 *
 * @param M Length of blocked M
 * @param N Length of blocked N
 * @param K Length of blocked K
 * @param sa Starting address of blocked A
 * @param sb Starting address of blocked B
 * @param sc Starting address of blocked C
 * @param ldc Leading dimension of original matrix C
 */
template <typename T>
void hgemm_kernel_1x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, T *sc, unsigned int ldc);
