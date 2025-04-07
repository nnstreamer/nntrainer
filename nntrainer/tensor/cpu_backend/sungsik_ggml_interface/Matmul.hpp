//
// Created by Rongjie Yi on 23-10-24.
//

#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP

#include "VecDot.hpp"

ErrorCode mat_mul(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = true, int thread_count = 4);

#endif // MLLM_MATMUL_HPP
