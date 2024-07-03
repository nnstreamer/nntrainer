// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.cpp
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <blas_kernels.h>

namespace nntrainer {
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  RunLayerContext &context, bool trans, bool trans_m) {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < input.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = input.getBatchSlice(b, 1);
    Tensor m_b = m.getBatchSlice(b, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    dotCl(this_b, m_b, result_b, context, trans, trans_m);
  }
}

Tensor dotCl(Tensor const &input, Tensor const &m, RunLayerContext &context,
             bool trans, bool trans_m) {
  Tensor output("", input.getFormat(), input.getDataType());
  dotCl(input, m, output, context, trans, trans_m);

  return output;
}

void dotCl(Tensor const &input, Tensor const &m, Tensor &result,
           RunLayerContext &context, bool trans, bool trans_m) {
  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = m.batch() * m.height() * m.width();
    mdim2 = m.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = m.batch() * m.channel() * m.height();
    mdim2 = m.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(),
                           input.getTensorType()); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(), input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = m.getData();
    float *rdata = result.getData();
    enum CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
    enum CBLAS_TRANSPOSE transB = trans_m ? CblasTrans : CblasNoTrans;

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K, context) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      transA ? sgemv_cl(data, mdata, rdata, dim2, dim1, lda, context)
             : sgemv_cl(data, mdata, rdata, dim1, dim2, lda, context);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      transB = transB == CblasTrans ? CblasNoTrans : CblasTrans;
      transB ? sgemv_cl(mdata, data, rdata, mdim2, mdim1, ldb, context)
             : sgemv_cl(mdata, data, rdata, mdim1, mdim2, ldb, context);
    }
    /// case others: use gemm
    else {
      // transA == false, transB == false
      sgemm_cl(data, mdata, rdata, M, N, K, lda, ldb, ldc, context);
      // todo: other condition implementations
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = input.getData<_FP16>();
    const _FP16 *mdata = m.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    enum CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
    enum CBLAS_TRANSPOSE transB = trans_m ? CblasTrans : CblasNoTrans;

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K, context) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      transA ? sgemv_cl(data, mdata, rdata, dim2, dim1, lda, context)
             : sgemv_cl(data, mdata, rdata, dim1, dim2, lda, context);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      transB = transB == CblasTrans ? CblasNoTrans : CblasTrans;
      transB ? sgemv_cl(mdata, data, rdata, mdim2, mdim1, ldb, context)
             : sgemv_cl(mdata, data, rdata, mdim1, mdim2, ldb, context);
    }
    /// case others: use sgemm
    else {
      // transA == false, transB == false
      sgemm_cl(data, mdata, rdata, M, N, K, lda, ldb, ldc, context);
      // todo: other condition implementations
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void multiplyCl(Tensor &input, float const &value, RunLayerContext &context) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData<float>();
    unsigned int len = input.size();

    sscal_cl(data, len, value, context);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data = input.getData<_FP16>();
    unsigned int len = input.size();
    sscal_cl(data, len, value, context);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void add_i_cl(Tensor const &input, Tensor &result, RunLayerContext &context) {

  CREATE_IF_EMPTY_DIMS(result, result.getDim());

  NNTR_THROW_IF(result.getData() == nullptr, std::invalid_argument)
    << result.getName() << " is not allocated";
  NNTR_THROW_IF(input.getData() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";

  if (input.getDim() != result.getDim()) {
    throw std::invalid_argument(
      "Error: Dimensions does not match for addition");
  }

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    unsigned int size = input.size();
    const float *data = input.getData();
    float *rdata = result.getData();

    addition_cl(data, rdata, size, context);

  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    unsigned int size = input.size();
    const _FP16 *data = input.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();

    addition_cl(data, rdata, size, context);

#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

} // namespace nntrainer
