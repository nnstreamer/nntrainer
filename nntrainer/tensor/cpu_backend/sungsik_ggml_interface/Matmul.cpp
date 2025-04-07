//
// Created by Rongjie Yi on 23-10-24.
//

#include "Matmul.hpp"
#include "Types.hpp"
#include "VecDotType.hpp"
#include <cassert>


ErrorCode mat_mul(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias,
                  bool transpose0, bool transpose1, int thread_count) {
    // src1 = W  src0 = x
    // transpose0=false  transpose1=true
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();

    auto src0_dtype = src0->dtype();
    auto src1_dtype = src1->dtype();
    auto vec_dot_type = type_traits[src1_dtype].vec_dot_type;
    auto vec_dot = type_traits[src1_dtype].vec_dot;
    auto x_to_vec_dot_type = type_traits[vec_dot_type].from_float;
    auto from_float_to_mat = type_traits[vec_dot_type].from_float_to_mat;
    mllm_gemv_func const gemv = type_traits[src1_dtype].gemv;
    mllm_gemm_func const gemm = type_traits[src1_dtype].gemm;
    auto blck_size_interleave = type_traits[src1_dtype].blck_size_interleave;

    auto src1_type_size = type_size(src1_dtype);
    auto src1_blck_size = blck_size(src1_dtype);
    auto src0_type_size = type_size(src0->dtype());
    auto src0_blck_size = blck_size(src0->dtype());


    auto not_vec_dot_type = src0_dtype != vec_dot_type;
    std::unique_ptr<Tensor> to; // later this tensor will be freed by ~Tensor
    if (not_vec_dot_type) {
        // convert x.dtype to vec_dot_type
        // so that we can use vec_dot to calculate dot product
        assert(src0_dtype == MLLM_TYPE_F32); // x should be fp32
        to = std::make_unique<Tensor>(src0->shape());
        to->setBackend(src0->backend());
        to->setDtype(vec_dot_type);
        to->alloc();
        to->setName(src0->name() + "-vec_dot");
        int64_t i_processed = 0;
        if ((from_float_to_mat != nullptr) && (gemv != nullptr) && dst->masterTensor() == nullptr) {
            for (int b = 0; b < src0->batch(); b++) {
                for (int h = 0; h < src0->head(); h++) {
#pragma omp parallel for collapse(1) num_threads(thread_count)
                    for (int64_t s = 0; s < src0->sequence() - src0->sequence() % 4; s += 4) {
                        from_float_to_mat(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                          (char *)to->rawHostPtr()
                                              + to->offset(b, h, s, 0) * type_size(to->dtype())
                                                    / blck_size(to->dtype()),
                                          4, src0->dimension(), blck_size_interleave);
                    }
                    i_processed = src0->sequence() - src0->sequence() % 4;
                }
            }
        }
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                for (int s = i_processed; s < src0->sequence(); s++) {
                    x_to_vec_dot_type(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                      (char *)to->rawHostPtr()
                                          + to->offset(b, h, s, 0) * type_size(to->dtype())
                                                / blck_size(to->dtype()),
                                      src0->dimension());
                }
            }
        }
        src0 = to.get();
        src0_dtype = src0->dtype();
        src0_type_size = type_size(src0->dtype());
        src0_blck_size = blck_size(src0->dtype());
    }

    if ((gemv != nullptr) && dst->dtypeAt(0, 0, 0, 0) == MLLM_TYPE_F32) {
        int nth = thread_count;
        if (!support_bias) {
#pragma omp parallel for collapse(1) num_threads(thread_count)
            for (int ith = 0; ith < nth; ith++) {
                int64_t i_processed = 0;
                int64_t seq_start = (ith * N) / nth; // = 0
                int64_t seq_end = ((ith + 1) * N) / nth; // = N
                if ((gemm != nullptr) && (M > 3) && dst->masterTensor() == nullptr) {
                    gemm(K, dst->hostPtr<float>() + dst->offset(0, 0, 0, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr(), M - M % 4, N / nth, /*bias=*/nullptr);
                    i_processed = M - M % 4; // if M is multiple of 4, ends here.
                }
                for (int iter = i_processed; iter < M; iter++) { // M-M%4
                    gemv(K, dst->hostPtr<float>() + dst->offset(0, 0, iter, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr()
                             + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                         1, N / nth, /*bias=*/nullptr);
                }
            }
        } else {
#pragma omp parallel for collapse(1) num_threads(thread_count)
            for (int ith = 0; ith < nth; ith++) {
                int64_t i_processed = 0;
                int64_t seq_start = (ith * N) / nth;
                int64_t seq_end = ((ith + 1) * N) / nth;
                if ((gemm != nullptr) && (M > 3) && dst->masterTensor() == nullptr) {
                    gemm(K, dst->hostPtr<float>() + dst->offset(0, 0, 0, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr(), M - M % 4, N / nth,
                         /*bias=*/bias->hostPtr<float>()
                             + bias->offset(/*b=*/0, /*h=*/0, /*s=*/0, /*d=*/seq_start));
                    i_processed = M - M % 4;
                }
                for (int iter = i_processed; iter < M; iter++) { // M-M%4
                    gemv(K, dst->hostPtr<float>() + dst->offset(0, 0, iter, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr()
                             + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                         1, N / nth,
                         /*bias=*/bias->hostPtr<float>()
                             + bias->offset(/*b=*/0, /*h=*/0, /*s=*/0, /*d=*/seq_start));
                }
            }
        }
        if (not_vec_dot_type) to->free();
        return MLLM_NO_ERROR;
    }
    return MLLM_NO_ERROR;
}
