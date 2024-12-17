// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	fused_fc_norm_cl.cpp
 * @date	7 May 2024
 * @brief	This is Fully Connected Layer Class for Neural Network with OpenCl
 * implementation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <common_properties.h>
#include <fused_fc_norm_cl.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FC_RMSParams { weight, bias, gamma };
// enum FCParams { weight, bias };
// enum RMSParams { gamma };

// , fc_rms_props(props::Unit(), props::FUSED_FC_RMS_NORM_GAMMA_INIT_GPU(),
// props::Epsilon())

FullyConnectedRMSNormLayerCl::FullyConnectedRMSNormLayerCl() :
  LayerImpl(),
  fc_rms_props(props::Unit(), props::FUSED_FC_RMS_NORM_GAMMA_INIT_GPU(),
               props::Epsilon()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedRMSNormLayerCl::finalize(InitLayerContext &context) {
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

  auto unit = std::get<props::Unit>(fc_rms_props).get();

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

  weight_idx[FC_RMSParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FC_RMSParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }

  // for RMS layer, size of output already set for fc, line 70
  auto &rmsparams_gamma =
    std::get<props::FUSED_FC_RMS_NORM_GAMMA_INIT_GPU>(fc_rms_props);

  TensorDim gamma_dim(
    1, 1, 1, output_dims[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  weight_idx[FC_RMSParams::gamma] =
    context.requestWeight(gamma_dim, rmsparams_gamma, WeightRegularizer::NONE,
                          1.0f, 0.0f, "gamma", false);
}

// TO-DO
/////////////////////////////////////////////////////////////////////////
// fc
void FullyConnectedRMSNormLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_rms_props, method, this);
}

void FullyConnectedRMSNormLayerCl::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_rms_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedRMSNormLayerCl::forwarding(RunLayerContext &context,
                                              bool training) {

  // for fc layer
  Tensor &weight = context.getWeight(weight_idx[FC_RMSParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  // for rms
  Tensor &gamma = context.getWeight(weight_idx[FC_RMSParams::gamma]);
  auto &epsilon = std::get<props::Epsilon>(fc_rms_props).get();

  auto disable_bias = std::get<props::DisableBias>(*layer_impl_props);
  bool disable_bias_value = disable_bias.empty() || disable_bias.get() == false;
  const Tensor &bias = context.getWeight(weight_idx[FC_RMSParams::bias]);
  // printf("\n*************************************************************************************************************************************\n");
  // printf("Bias value : %s\n", disable_bias_value ? "true" : "false");
  // printf("\nInput Tensor Batch: %u, Channel: %u, Height: %u, Width: %u\n",
  // input_.batch(), input_.channel(), input_.height(), input_.width()); for
  // (unsigned int i = 0; i < input_.size(); ++i) {
  //   printf("Element %u -> %f\n", i, *(input_.getData<float>() + i));
  // }
  // printf("\nWeight Tensor Batch: %u, Channel: %u, Height: %u, Width: %u\n",
  // weight.batch(), weight.channel(), weight.height(), weight.width());
  // printf("\nHidden Tensor Batch: %u, Channel: %u, Height: %u, Width: %u\n",
  // hidden_.batch(), hidden_.channel(), hidden_.height(), hidden_.width());
  // printf("\nGamma Tensor Batch: %u, Channel: %u, Height: %u, Width: %u\n",
  // gamma.batch(), gamma.channel(), gamma.height(), gamma.width());
  // printf("\nEpsilon value : %f\n", epsilon);
  // printf("\n-----------------------------------------starting with fusion
  // process from layer side-----------------------------------------------\n");

  fusedProcess(input_, weight, hidden_, bias, disable_bias_value, gamma,
               epsilon);
}

// TO-DO
////// need to implement the incremental forwarding
void FullyConnectedRMSNormLayerCl::incremental_forwarding(
  RunLayerContext &context, unsigned int from, unsigned int to, bool training) {
  Tensor w;
  Tensor &weight = w;
  context.getWeight(weight, weight_idx[FC_RMSParams::weight]);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  // rms
  Tensor &gamma = context.getWeight(weight_idx[FC_RMSParams::gamma]);

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

  // @todo: set reset stride as false. This implementation only works when
  // batch size is 1
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);
  Tensor hidden_step = hidden_.getSharedDataTensor(hidden_step_dim, 0, true);

  auto &epsilon = std::get<props::Epsilon>(fc_rms_props).get();

  auto disable_bias = std::get<props::DisableBias>(*layer_impl_props);
  bool disable_bias_value = disable_bias.empty() || disable_bias.get() == false;
  Tensor &bias = context.getWeight(weight_idx[FC_RMSParams::bias]);

  fusedProcess(input_step, weight, hidden_step, bias, disable_bias_value, gamma,
               epsilon);
}

void FullyConnectedRMSNormLayerCl::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FC_RMSParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void FullyConnectedRMSNormLayerCl::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FC_RMSParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FC_RMSParams::bias]);

    if (context.isGradientFirstAccess(weight_idx[FC_RMSParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      /// @todo optimize below by adding beta to Tensor::sum
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FC_RMSParams::weight]));
}

} /* namespace nntrainer */
