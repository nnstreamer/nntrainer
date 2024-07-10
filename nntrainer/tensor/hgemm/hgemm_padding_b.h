// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_padding_b.h
 * @date   05 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header file for padding function used in hgemm
 *
 */

void hgemm_padding_B(const __fp16 *B, __fp16 *Bp, unsigned int K,
                     unsigned int N, unsigned int K8, unsigned int N16,
                     bool transB);

void hgemm_padding_B_noTrans(const __fp16 *B, __fp16 *Bp, unsigned int K,
                             unsigned int N, unsigned int K8, unsigned int N16);

void hgemm_padding_B_noTrans_wrt_N(const __fp16 *B, __fp16 *Bp,
                                   unsigned int K, unsigned int N,
                                   unsigned int K8, unsigned int N16);

void hgemm_padding_B_noTrans_wrt_K(const __fp16 *B, __fp16 *Bp,
                                   unsigned int K, unsigned int N,
                                   unsigned int K8, unsigned int N16);

void hgemm_padding_B_noTrans_wrt_KN(const __fp16 *B, __fp16 *Bp,
                                    unsigned int K, unsigned int N,
                                    unsigned int K8, unsigned int N16);

void hgemm_padding_B_Trans(const __fp16 *B, __fp16 *Bp, unsigned int K,
                           unsigned int N, unsigned int K8, unsigned int N16);

void hgemm_padding_B_Trans_wrt_N(const __fp16 *B, __fp16 *Bp,
                                 unsigned int K, unsigned int N,
                                 unsigned int K8, unsigned int N16);

void hgemm_padding_B_Trans_wrt_K(const __fp16 *B, __fp16 *Bp,
                                 unsigned int K, unsigned int N,
                                 unsigned int K8, unsigned int N16);

void hgemm_padding_B_Trans_wrt_KN(const __fp16 *B, __fp16 *Bp,
                                  unsigned int K, unsigned int N,
                                  unsigned int K8, unsigned int N16);
