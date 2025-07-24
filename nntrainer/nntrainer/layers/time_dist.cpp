// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   time_dist.cpp
 * @date   01 April 2021
 * @brief  This is Time Distributed Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <time_dist.h>
#include <util_func.h>
#include <weight.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

static void reshape(Tensor &m) {
  TensorDim d = m.getDim();
  m.reshape({d[2], d[1], d[0], d[3]});
}

void TimeDistLayer::setPosition(RunLayerContext &context) {
  positions[0] = context.getInput(SINGLE_INOUT_IDX).getData();
  positions[2] = context.getOutput(SINGLE_INOUT_IDX).getData();
  /** TODO: use mode of execution here */
  try {
    positions[1] = context.getOutgoingDerivative(SINGLE_INOUT_IDX).getData();
    positions[3] =
      (float *)context.getIncomingDerivative(SINGLE_INOUT_IDX).getData();
  } catch (...) {
    /** in case of training, these tensors will not exist */
  }
}

void TimeDistLayer::transposeInOut(RunLayerContext &context) {
  // Position[0] : net_input.variable
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  input_.copy(transposeTensor(input_));

  // Position[1] : net_input.gradient
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  if (ret_.getData() != positions[0]) {
    ret_.copy(transposeTensor(ret_));
  } else {
    reshape(ret_);
  }

  // Position[2] : net_hidden.variable
  Tensor &hval_ = context.getOutput(SINGLE_INOUT_IDX);
  if (hval_.getData() != positions[0] && hval_.getData() != positions[1]) {
    hval_.copy(transposeTensor(hval_));
  } else {
    reshape(hval_);
  }

  // Position[3] : net_hidden.gradient
  bool trans = true;

  /// @fixme: below will be propably wrong as this changes incoming derivative.
  /// other layer referring to this will have wrong output grad information.
  Tensor &derivative_ = context.getOutputGradUnsafe(SINGLE_INOUT_IDX);
  for (unsigned int i = 0; i < 3; ++i) {
    if (derivative_.getData() == positions[i]) {
      trans = false;
      break;
    }
  }
  if (trans)
    derivative_.copy(transposeTensor(derivative_));
  else
    reshape(derivative_);
}

Tensor TimeDistLayer::transposeTensor(Tensor &m) {
  TensorDim dim = m.getDim();
  // Assume the channel is 1. Time Dimension is h. It transpose [b, 1, h, w] to
  // [h, 1, b, w ] and nntrainer only support 1,2,3 transpose. So we do reshape
  // first to make [1, b,h, w]
  // TODO:
  // If we do {1, dim[0]*dim[1], dim[2], dim[3]} and transpose to {1, dim[2],
  // dim[0]*dim[1], dim[3]}. Then reshpae to {dim[2], dim[0], dim[1], dim[3]}
  // then we could support the case which dim[1] is not 1. But we need to change
  // some other places of code to support.
  //
  if (dim[1] != 1)
    throw std::invalid_argument(
      "Channel of Time distributed layer must be 1 for now");

  m.reshape({dim[1], dim[0], dim[2], dim[3]});
  Tensor in = m.transpose("1:0:2");
  in.reshape({dim[2], dim[1], dim[0], dim[3]});
  m.reshape(dim);
  in.setName(m.getName() + "_trans");

  return in;
}

void TimeDistLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Time distributed layer takes only one input";

  if (!dist_layer) {
    throw std::invalid_argument("distributed layer is not set properly");
  }

  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.channel() != 1) {
    throw std::invalid_argument(
      "only 1 channel is allow for time distributed layer");
  }

  /**
   * simulate an InitLayerContext, and then replicate its effect onto the
   * actual context
   */
  TensorDim dist_dim = input_dim;
  dist_dim.height(1);
  InitLayerContext dist_context({dist_dim}, {}, context.getInPlace(),
                                context.getName());

  // During forwarding and backwarding, it set the input and output buffer of
  // dist_layer properly
  // dist_layer will use forwarding_with_val and backwarding_with_val
  dist_layer->finalize(dist_context);

  TensorDim output_dim = dist_context.getOutSpecs()[0].variable_spec.dim;
  // input_dim.height is number of time iteration
  output_dim.height(input_dim.height());
  context.setOutputDimensions({output_dim});

  /** real setting of context */
  fillLayerInitContext(context, dist_context);
}

void TimeDistLayer::fillWeightsFromContext(RunLayerContext &context) {
  weights_wrapper.resize(context.getNumWeights());

  /** create weights */
  for (unsigned int idx = 0; idx < context.getNumWeights(); idx++) {
    if (context.weightHasGradient(idx)) {
      weights_wrapper[idx] =
        Weight(context.getWeight(idx), context.getWeightGrad(idx),
               context.getWeightName(idx));
    } else {
      weights_wrapper[idx] =
        Weight(context.getWeight(idx), Tensor(), context.getWeightName(idx));
    }
  }
}

void TimeDistLayer::fillTensorsFromContext(RunLayerContext &context) {
  tensors_wrapper.resize(context.getNumTensors());

  /** create tensors */
  for (unsigned int idx = 0; idx < context.getNumTensors(); idx++) {
    if (context.tensorHasGradient(idx)) {
      tensors_wrapper[idx] =
        Var_Grad(context.getTensor(idx), context.getTensorGrad(idx),
                 context.getTensorName(idx));
    } else {
      tensors_wrapper[idx] =
        Var_Grad(context.getTensor(idx), Tensor(), context.getTensorName(idx));
    }
  }
}

std::vector<Weight *> TimeDistLayer::getWeightsForContext() {
  /** create weights for context */
  std::vector<Weight *> weights_for_context;
  for (auto &w : weights_wrapper)
    weights_for_context.push_back(&w);

  return weights_for_context;
}

std::vector<Var_Grad *> TimeDistLayer::getTensorsForContext() {
  /** create tensors for context */
  std::vector<Var_Grad *> tensors_for_context;
  for (auto &t : tensors_wrapper)
    tensors_for_context.push_back(&t);

  return tensors_for_context;
}

void TimeDistLayer::forwarding(RunLayerContext &context, bool training) {
  setPosition(context);

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  // input_.dim = [ b, 1, h, w ]

  Tensor h_g;

  const TensorDim &ho_dim = hidden_.getDim();
  const TensorDim &in_dim = input_.getDim();

  // TODO: This transposed Input Tensor could be resued for backwarding
  Tensor in = transposeTensor(input_);

  Tensor out = Tensor({ho_dim[2], 1, ho_dim[0], ho_dim[3]}, true,
                      Initializer::NONE, context.getName() + ":inter_output");

  TensorDim i_dim = in_dim;
  i_dim.channel(1);
  i_dim.height(1);

  TensorDim h_dim = ho_dim;
  h_dim.channel(1);
  h_dim.height(1);

  if (dist_layer->requireLabel() &&
      context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    Tensor &hidden_g = context.getLabel(SINGLE_INOUT_IDX);
    h_g = transposeTensor(hidden_g);
  }

  Var_Grad in_var(i_dim, Initializer::NONE, false, false, "input");
  Var_Grad out_var(h_dim, Initializer::NONE,
                   dist_layer->requireLabel() &&
                     context.isLabelAvailable(SINGLE_INOUT_IDX),
                   false, "output");

  fillWeightsFromContext(context);
  fillTensorsFromContext(context);

  for (unsigned int i = 0; i < in_dim.height(); ++i) {
    //
    // Iterate Height Direction. The dimension of in is input_[ b, 1, 1, width
    // ]. The dimension of out is hidden_[ b, 1, 1, width ]
    //
    Tensor label_iter;

    Tensor in_iter = in.getSharedDataTensor(
      i_dim, i * in_dim.batch() * in_dim.width(), true, in.getName());
    Tensor out_iter = out.getSharedDataTensor(
      h_dim, i * ho_dim.batch() * ho_dim.width(), true, out.getName());

    in_var.initializeVariable(in_iter);
    out_var.initializeVariable(out_iter);

    if (dist_layer->requireLabel() &&
        context.isLabelAvailable(SINGLE_INOUT_IDX)) {
      label_iter = h_g.getSharedDataTensor(
        h_dim, i * ho_dim.batch() * ho_dim.width(), true, h_g.getName());
      out_var.initializeGradient(label_iter);
    }

    RunLayerContext dist_context(
      context.getName(), context.getTrainable(), context.getLoss(),
      context.getInPlace(), context.getLossScale(), context.getContextData(),
      false, getWeightsForContext(), {&in_var}, {&out_var},
      getTensorsForContext());

    dist_layer->forwarding(dist_context, training);
  }

  hidden_.copy(transposeTensor(out));
  clearFromContext();
}

void TimeDistLayer::calcDerivative(RunLayerContext &context) {
  /// @fixme: this will be probably wrong as this mutates incoming derivative,
  /// we will need the layer to copy and paste instead of transpose and override
  Tensor &derivative_ = context.getOutputGradUnsafe(SINGLE_INOUT_IDX);
  Tensor &hval_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  TensorDim der_dim = derivative_.getDim();
  TensorDim ret_dim = ret_.getDim();

  TensorDim r_dim = {ret_dim[2], 1, 1, ret_dim[3]};
  TensorDim d_dim = {der_dim[2], 1, 1, der_dim[3]};

  Var_Grad in_var(r_dim, Initializer::NONE, true, false, "input");
  Var_Grad out_var(d_dim, Initializer::NONE, true, false, "output");

  fillWeightsFromContext(context);
  fillTensorsFromContext(context);

  for (unsigned int i = 0; i < der_dim[0]; ++i) {
    Tensor ret_iter = ret_.getSharedDataTensor(
      r_dim, i * r_dim.batch() * r_dim.width(), true, ret_.getName());
    Tensor in_iter = input_.getSharedDataTensor(
      r_dim, i * r_dim.batch() * r_dim.width(), true, input_.getName());
    Tensor d_iter = derivative_.getSharedDataTensor(
      d_dim, i * d_dim.batch() * d_dim.width(), true, derivative_.getName());
    Tensor hval_iter = hval_.getSharedDataTensor(
      d_dim, i * d_dim.batch() * d_dim.width(), true, hval_.getName());

    in_var.initializeGradient(ret_iter);
    in_var.initializeVariable(in_iter);
    out_var.initializeGradient(d_iter);
    out_var.initializeVariable(hval_iter);

    RunLayerContext dist_context(
      context.getName(), context.getTrainable(), context.getLoss(),
      context.getInPlace(), context.getLossScale(), context.getContextData(),
      false, getWeightsForContext(), {&in_var}, {&out_var},
      getTensorsForContext());

    dist_layer->calcDerivative(dist_context);
  }

  ret_.copy(transposeTensor(ret_));
  // We are not going to transpose the data. The Date is not used anymore.
  // It will be overwritten at next iteration
  // Just reshpae the tensors
  hval_.reshape({der_dim[2], 1, der_dim[0], der_dim[3]});
  derivative_.reshape({der_dim[2], 1, der_dim[0], der_dim[3]});
  input_.reshape({ret_dim[2], 1, ret_dim[0], ret_dim[3]});
  clearFromContext();
}

void TimeDistLayer::calcGradient(RunLayerContext &context) {
  // Even if the dist_layer->getNumWeights() == 0, We do transpose here
  // for the calculation of derivatives and overwrite original tensors.
  // And use them in calcDerivatives() without transpose.
  transposeInOut(context);

  if (context.getNumWeights() == 0)
    return;

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  TensorDim der_dim = derivative_.getDim();
  TensorDim in_dim = input_.getDim();

  TensorDim i_dim = {in_dim[2], 1, 1, in_dim[3]};
  TensorDim d_dim = {der_dim[2], 1, 1, der_dim[3]};

  fillWeightsFromContext(context);
  fillTensorsFromContext(context);

  for (unsigned int i = 0; i < der_dim[0]; ++i) {
    Tensor in_iter = input_.getSharedDataTensor(
      i_dim, i * i_dim.batch() * i_dim.width(), true, input_.getName());
    Tensor d_iter = derivative_.getSharedDataTensor(
      d_dim, i * d_dim.batch() * d_dim.width(), true, derivative_.getName());

    Var_Grad in_var(i_dim, Initializer::NONE, true, false, "input");
    Var_Grad out_var(d_dim, Initializer::NONE, true, false, "output");

    in_var.initializeVariable(in_iter);
    out_var.initializeGradient(d_iter);

    RunLayerContext dist_context(
      context.getName(), context.getTrainable(), context.getLoss(),
      context.getInPlace(), context.getLossScale(), context.getContextData(),
      false, getWeightsForContext(), {&in_var}, {&out_var},
      getTensorsForContext());

    dist_layer->calcGradient(dist_context);
  }
  clearFromContext();
}

void TimeDistLayer::fillLayerInitContext(InitLayerContext &context,
                                         const InitLayerContext &dist_context) {
  /** real set the input flags */
  auto const &input_dims = context.getInputDimensions();
  for (unsigned int idx = 0; idx < dist_context.getNumInputs(); idx++) {
    context.setDynDimFlagInputDimension(idx, input_dims[idx].getDynDimFlag());
    context.setEffDimFlagInputDimension(idx, input_dims[idx].getEffDimFlag());
  }

  /** real request of tensors */
  for (auto const &ts : dist_context.getTensorsSpec())
    context.requestTensor(ts);

  /** real request of weights */
  for (auto const &ws : dist_context.getWeightsSpec())
    context.requestWeight(ws);
}

void TimeDistLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  if (context.getNumTensors() > 0) {
    const TensorDim &out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();
    const TensorDim &in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();

    TensorDim i_dim = {in_dim[2], 1, 1, in_dim[3]};
    TensorDim o_dim = {out_dim[2], 1, 1, out_dim[3]};

    Var_Grad in_var(i_dim, Initializer::NONE, true, false, "input");
    Var_Grad out_var(o_dim, Initializer::NONE, true, false, "output");

    fillWeightsFromContext(context);
    fillTensorsFromContext(context);

    RunLayerContext dist_context(
      context.getName(), context.getTrainable(), context.getLoss(),
      context.getInPlace(), context.getLossScale(), context.getContextData(),
      false, getWeightsForContext(), {&in_var}, {&out_var},
      getTensorsForContext());

    dist_layer->setBatch(dist_context, batch);

    for (unsigned int idx = 0; idx < dist_context.getNumTensors(); idx++) {
      context.updateTensor(idx, dist_context.getTensor(idx).getDim().batch());
    }

    clearFromContext();
  }
}

} /* namespace nntrainer */
