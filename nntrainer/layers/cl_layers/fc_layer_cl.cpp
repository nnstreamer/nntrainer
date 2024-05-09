// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	fc_layer_cl.cpp
 * @date	7 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network with OpenCl
 * implementation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <fc_layer_cl.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

std::string fc_sgemv_cl_kernel_ =
  R"(__kernel void fc_sgemv_cl(const __global float* A, const __global float* X,
                      __global float* Y, unsigned int M, unsigned int N) {                                            
        unsigned int i, j;
        i = get_global_id(0);                         
        float y0 = Y[i] * 0.0f;
        for (unsigned int j = 0; j < M; j++)                         
            y0 += A[i + j * N] * X[j]; 
        Y[i] = y0;                            
          
    })";

std::string fc_dot_cl_kernel_ =
  R"(__kernel void fc_dot_cl(const __global float* A, const __global float* X, unsigned int K, float res) {
        res = 0;
        for (unsigned int i = 0; i < K; i++){
            res += A[i] * X[i];
        }
    })";

std::string fc_sgemm_cl_kernel_ =
  R"(__kernel void fc_sgemm_cl(const __global float* A, const __global float* B,
                      __global float* C, unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        for (unsigned int n = 0; n < N; ++n) {
          float c = 0.0;
          float c_old = C[m * ldc + n];
          for (unsigned int k = 0; k < K; ++k) {
            float a, b;
            a = A[m * lda + k];
            b = B[k * ldb + n];
            c += a * b;
          }
          C[m * ldc + n] = c;
        }
    })";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };

FullyConnectedLayerCl::FullyConnectedLayerCl() :
  LayerImpl(), fc_props(props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedLayerCl::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(fc_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }
}

void FullyConnectedLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayerCl::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayerCl::forwarding(RunLayerContext &context,
                                       bool training) {

  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (weight.getDataType() == nntrainer::Tdatatype::QINT4 ||
      weight.getDataType() == nntrainer::Tdatatype::QINT8) {
    Tdatatype dtype = input_.getDataType();

    Tensor weight_(
      {{weight.batch(), weight.channel(), weight.height(), weight.width()},
       {weight.getFormat(), dtype}},
      true);

    unsigned int axis =
      context.getWeightObject(weight_idx[FCParams::weight]).getOutputAxis();

    weight.dequantize(weight_, axis);

    fcDotProcess(input_, weight_, hidden_, context);
  } else {
    fcDotProcess(input_, weight, hidden_, context);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

/**
 * @brief declaring static kernel objects
 *
 */
opencl::Kernel FullyConnectedLayerCl::kernel_sgemv;
opencl::Kernel FullyConnectedLayerCl::kernel_sgemm;
opencl::Kernel FullyConnectedLayerCl::kernel_dot;

void FullyConnectedLayerCl::fcDotProcess(Tensor const &input,
                                         Tensor const &weight, Tensor &result,
                                         RunLayerContext &context) {
  // to do:
  // NNTR_THROW_IF(!contiguous, std::invalid_argument)
  //   << getName() << " is not contiguous. Cannot dot product.";

  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = weight.batch() * weight.height() * weight.width();
    mdim2 = weight.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = weight.batch() * weight.channel() * weight.height();
    mdim2 = weight.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;
  if (dim2 != mdim1)
    throw std::runtime_error("Error: incompatible dimensions for dot product");
  K = mdim1; /** == dim2 */
  N = mdim2;
  M = dim1;
  if (input.getFormat() == Tformat::NHWC) {
    CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                         input.width(),
                         input.getTensorType()); //  NHWC Result Tensor
  } else {
    CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                         N, input.getTensorType());
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = weight.getData();
    float *rdata = result.getData();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = fc_dot_cl(data, mdata, K, context) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      fc_sgemv_cl(data, mdata, rdata, dim1, dim2, lda, context);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      fc_sgemv_cl(mdata, data, rdata, mdim1, mdim2, ldb, context);
    }
    /// case others: use gemm
    else {
      fc_sgemm_cl(data, mdata, rdata, M, N, K, lda, ldb, ldc, context);
    }
  } else
    throw std::invalid_argument("Error: OpenCL fp16 is not supported yet.");
}

void FullyConnectedLayerCl::fc_sgemv_cl(const float *matAdata,
                                        const float *vecXdata, float *vecYdata,
                                        unsigned int dim1, unsigned int dim2,
                                        unsigned int lda,
                                        RunLayerContext &context) {

  bool result = false;

  do {
    result =
      context.clCreateKernel(fc_sgemv_cl_kernel_, context.LayerKernel::FCSGEMV,
                             FullyConnectedLayerCl::kernel_sgemv);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    opencl::Buffer inputA(context.context_inst_, dim1_size * dim2_size, true,
                          nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutY(context.context_inst_, dim2_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemv.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemv.SetKernelArguments(
      1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemv.SetKernelArguments(
      2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemv.SetKernelArguments(
      3, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemv.SetKernelArguments(
      4, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      FullyConnectedLayerCl::kernel_sgemv, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

float FullyConnectedLayerCl::fc_dot_cl(const float *matAdata,
                                       const float *vecXdata, unsigned int dim1,
                                       RunLayerContext &context) {

  bool result = false;

  float cl_ret = 0;

  do {
    // FullyConnectedLayerCl::kernel_ is wrong for this ...its sgemv.
    result =
      context.clCreateKernel(fc_dot_cl_kernel_, context.LayerKernel::FCDOT,
                             FullyConnectedLayerCl::kernel_dot);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;

    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_dot.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_dot.SetKernelArguments(
      1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_dot.SetKernelArguments(2, &dim1,
                                                                  sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_dot.SetKernelArguments(
      3, &cl_ret, sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      FullyConnectedLayerCl::kernel_dot, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void FullyConnectedLayerCl::fc_sgemm_cl(const float *A, const float *B,
                                        float *C, unsigned int M,
                                        unsigned int N, unsigned int K,
                                        unsigned int lda, unsigned int ldb,
                                        unsigned int ldc,
                                        RunLayerContext &context) {

  bool result = false;

  do {
    result =
      context.clCreateKernel(fc_sgemm_cl_kernel_, context.LayerKernel::FCSGEMM,
                             FullyConnectedLayerCl::kernel_sgemm);
    if (!result) {
      break;
    }

    size_t m_size = sizeof(float) * M;
    size_t n_size = sizeof(float) * N;
    size_t k_size = sizeof(float) * K;
    opencl::Buffer inputA(context.context_inst_, m_size * k_size, true,
                          nullptr);

    opencl::Buffer inputB(context.context_inst_, k_size * n_size, true,
                          nullptr);

    opencl::Buffer inOutC(context.context_inst_, m_size * n_size, true,
                          nullptr);

    result = inputA.WriteData(context.command_queue_inst_, A);
    if (!result) {
      break;
    }

    result = inputB.WriteData(context.command_queue_inst_, B);
    if (!result) {
      break;
    }

    result = inOutC.WriteData(context.command_queue_inst_, C);
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      3, &M, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      4, &N, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      5, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      6, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      7, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_sgemm.SetKernelArguments(
      8, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      FullyConnectedLayerCl::kernel_sgemm, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutC.ReadData(context.command_queue_inst_, C);
    if (!result) {
      break;
    }

  } while (false);
}

void FullyConnectedLayerCl::incremental_forwarding(RunLayerContext &context,
                                                   unsigned int from,
                                                   unsigned int to,
                                                   bool training) {
  Tensor w;
  Tensor &weight = w;
  context.getWeight(weight, weight_idx[FCParams::weight]);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  input_step_dim.height(to - from);
  hidden_step_dim.height(to - from);

  // @todo: set reset stride as false. This implementation only works when batch
  // size is 1
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);
  Tensor hidden_step = hidden_.getSharedDataTensor(hidden_step_dim, 0, true);

  fcDotProcess(input_step, weight, hidden_step, context);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_step.add_i(bias);
  }
}

void FullyConnectedLayerCl::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void FullyConnectedLayerCl::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);

    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      /// @todo optimize below by adding beta to Tensor::sum
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} /* namespace nntrainer */
