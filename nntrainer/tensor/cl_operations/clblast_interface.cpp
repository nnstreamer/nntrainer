// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file	clblast_interface.cpp
 * @date	12 May 2025
 * @brief	CLBlast library interface
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <stdexcept>

#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED // to disable deprecation warnings
#include "clblast.h"
#include "clblast_interface.h"

namespace nntrainer {

void scal_cl(const unsigned int N, const float alpha, float *X,
             unsigned int incX) {
  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Scal<float>(N, alpha, clBuffManagerInst.getOutBufferA()->GetBuffer(),
                       0, incX, &command_queue);

  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);
}

void copy_cl(const unsigned int N, const float *X, float *Y, unsigned int incX,
             unsigned int incY) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Copy<float>(N, clBuffManagerInst.getInBufferA()->GetBuffer(), 0,
                       incX, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       incY, &command_queue);

  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);
}

void axpy_cl(const unsigned int N, const float alpha, const float *X, float *Y,
             unsigned int incX, unsigned int incY) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);

  clblast::Axpy<float>(N, alpha, clBuffManagerInst.getInBufferA()->GetBuffer(),
                       0, incX, clBuffManagerInst.getOutBufferA()->GetBuffer(),
                       0, incY, &command_queue);

  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);
}

float dot_cl(const unsigned int N, const float *X, const float *Y,
             unsigned int incX, unsigned int incY) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clBuffManagerInst.getInBufferB()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), Y);

  clblast::Dot<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                      clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                      clBuffManagerInst.getInBufferB()->GetBuffer(), 0, incY,
                      &command_queue);

  float result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(float), &result);
  return result;
}

float nrm2_cl(const unsigned int N, const float *X, unsigned int incX) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Nrm2<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  float result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(float), &result);

  return result;
}

float asum_cl(const unsigned int N, const float *X, unsigned int incX) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Asum<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  float result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(float), &result);

  return result;
}

int amax_cl(const unsigned int N, const float *X, unsigned int incX) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Amax<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  int result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(int), &result);

  return result;
}

int amin_cl(const unsigned int N, const float *X, unsigned int incX) {
  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, N * sizeof(float), X);

  clblast::Amin<float>(N, clBuffManagerInst.getOutBufferA()->GetBuffer(), 0,
                       clBuffManagerInst.getInBufferA()->GetBuffer(), 0, incX,
                       &command_queue);

  int result;
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, sizeof(int), &result);

  return result;
}

void gemv_cl(const unsigned int layout, bool TransA, const unsigned int M,
             const unsigned int N, const float alpha, const float *A,
             const unsigned int lda, const float *X, const float beta, float *Y,
             unsigned int incX, unsigned int incY) {
  throw std::runtime_error("gemv_cl is not implemented");
}

void gemm_cl(const unsigned int layout, bool TransA, bool TransB,
             const unsigned int M, const unsigned int N, const unsigned int K,
             const float alpha, const float *A, const unsigned int lda,
             const float *B, const unsigned int ldb, const float beta, float *C,
             const unsigned int ldc) {
  clblast::Transpose transA =
    (TransA) ? clblast::Transpose::kYes : clblast::Transpose::kNo;
  clblast::Transpose transB =
    (TransB) ? clblast::Transpose::kYes : clblast::Transpose::kNo;

  clBuffManagerInst.getInBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, M * K * sizeof(float), A);

  clBuffManagerInst.getInBufferB()->WriteDataRegion(
    clblast_cc->command_queue_inst_, K * N * sizeof(float), B);

  clBuffManagerInst.getOutBufferA()->WriteDataRegion(
    clblast_cc->command_queue_inst_, M * N * sizeof(float), C);

  // layout is currently fixed to RowMajor
  clblast::Gemm<float>(
    clblast::Layout::kRowMajor, transA, transB, M, N, K, alpha,
    clBuffManagerInst.getInBufferA()->GetBuffer(), 0, lda,
    clBuffManagerInst.getInBufferB()->GetBuffer(), 0, ldb, beta,
    clBuffManagerInst.getOutBufferA()->GetBuffer(), 0, ldc, &command_queue);

  // Read the result back to C
  clBuffManagerInst.getOutBufferA()->ReadDataRegion(
    clblast_cc->command_queue_inst_, M * N * sizeof(float), C);
}

void gemm_batched_cl(const unsigned int layout, bool TransA, bool TransB,
                     const unsigned int M, const unsigned int N,
                     const unsigned int K, const float *alpha, const float *A,
                     const unsigned int lda, const float *B,
                     const unsigned int ldb, const float *beta, float *C,
                     const unsigned int ldc, const unsigned int batch_size) {
  throw std::runtime_error("gemm_batched_cl is not implemented");
}

void im2col_cl(const unsigned int C, const unsigned int H, const unsigned int W,
               const unsigned int kernel_h, const unsigned int kernel_w,
               const unsigned int pad_h, const unsigned int pad_w,
               const unsigned int stride_h, const unsigned int stride_w,
               const unsigned int dilation_h, const unsigned int dilation_w,
               const float *input, float *output) {
  throw std::runtime_error("im2col_cl is not implemented");
}

} // namespace nntrainer
