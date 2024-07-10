// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_padding_a.h
 * @date   05 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header file for padding function used in hgemm
 *
 */

void hgemm_padding_A(const __fp16 *A, __fp16 *Ap, unsigned int M,
                     unsigned int K, unsigned int M8, unsigned int K8,
                     bool transA);
void hgemm_padding_A_noTrans(const __fp16 *A, __fp16 *Ap, unsigned int M,
                             unsigned int K, unsigned int M8, unsigned int K8);
void hgemm_padding_A_noTrans_wrt_M(const __fp16 *A, __fp16 *Ap,
                                   unsigned int M, unsigned int K,
                                   unsigned int M8, unsigned int K8);
void hgemm_padding_A_noTrans_wrt_K(const __fp16 *A, __fp16 *Ap,
                                   unsigned int M, unsigned int K,
                                   unsigned int M8, unsigned int K8);
void hgemm_padding_A_noTrans_wrt_MK(const __fp16 *A, __fp16 *Ap,
                                    unsigned int M, unsigned int K,
                                    unsigned int M8, unsigned int K8);
void hgemm_padding_A_Trans(const __fp16 *A, __fp16 *Ap, unsigned int M,
                           unsigned int K, unsigned int M8, unsigned int K8);
void hgemm_padding_A_Trans_wrt_M(const __fp16 *A, __fp16 *Ap,
                                 unsigned int M, unsigned int K,
                                 unsigned int M8, unsigned int K8);
void hgemm_padding_A_Trans_wrt_K(const __fp16 *A, __fp16 *Ap,
                                 unsigned int M, unsigned int K,
                                 unsigned int M8, unsigned int K8);
void hgemm_padding_A_Trans_wrt_MK(const __fp16 *A, __fp16 *Ap,
                                  unsigned int M, unsigned int K,
                                  unsigned int M8, unsigned int K8);
