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

#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <time_dist.h>
#include <util_func.h>

namespace nntrainer {

const std::string TimeDistLayer::type = "time_dist";

static void reshape(Tensor &m) {
  TensorDim d = m.getDim();
  m.reshape({d[2], d[1], d[0], d[3]});
}

void TimeDistLayer::setPosition() {
  positions[0] = net_input[0]->getVariableRef().getData();
  positions[1] = net_input[0]->getGradientRef().getData();
  positions[2] = net_hidden[0]->getVariableRef().getData();
  positions[3] = net_hidden[0]->getGradientRef().getData();
}

void TimeDistLayer::transposeInOut() {
  // Position[0] : net_input.variable
  Tensor &input_ = net_input[0]->getVariableRef();
  input_.copy(transposeTensor(input_));

  // Position[1] : net_input.gradient
  Tensor &ret_ = net_input[0]->getGradientRef();
  if (ret_.getData() != positions[0]) {
    ret_.copy(transposeTensor(ret_));
  } else {
    reshape(ret_);
  }

  // Position[2] : net_hidden.variable
  Tensor &hval_ = net_hidden[0]->getVariableRef();
  if (hval_.getData() != positions[0] && hval_.getData() != positions[1]) {
    hval_.copy(transposeTensor(hval_));
  } else {
    reshape(hval_);
  }

  // Position[3] : net_hidden.gradient
  bool trans = true;

  Tensor &derivative_ = net_hidden[0]->getGradientRef();
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

  return in;
}

int TimeDistLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (getNumInputs() != 1) {
    throw std::invalid_argument("Time distributed layer takes only one input");
  }

  if (!dist_layer) {
    throw std::invalid_argument("distributed layer is not set properly");
  }

  if (input_dim[0].channel() != 1) {
    throw std::invalid_argument(
      "only 1 channel is allow for time distributed layer");
  }

  TensorDim dist_dim = input_dim[0];
  dist_dim.height(1);

  dist_layer->setInputDimension({dist_dim});

  // Set the weight of dist_layer
  // Input & Output Buffer is set by manager of model.
  // During forwarding and backwarding, it set the input and output buffer of
  // dist_layer properly
  // dist_layer will use forwarding_with_val and backwarding_with_val
  dist_layer->initialize(manager);

  output_dim[0] = dist_layer->getOutputDimension()[0];

  // input_dim[0].height is number of time iteration
  output_dim[0].height(input_dim[0].height());

  return status;
}

void TimeDistLayer::forwarding(bool training) {
  setPosition();

  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();
  // input_.dim = [ b, 1, h, w ]

  Tensor hidden_g, h_g;

  TensorDim ho_dim = hidden_.getDim();
  TensorDim in_dim = input_.getDim();

  // TODO: This transposed Input Tensor could be resued for backwarding
  Tensor in = transposeTensor(input_);

  Tensor out = Tensor({ho_dim[2], 1, ho_dim[0], ho_dim[3]}, true);

  TensorDim i_dim = in_dim;
  i_dim.channel(1);
  i_dim.height(1);

  TensorDim h_dim = ho_dim;
  h_dim.channel(1);
  h_dim.height(1);

  if (dist_layer->getType() == "loss") {
    hidden_g = net_hidden[0]->getGradientRef();
    if (!hidden_g.uninitialized()) {
      h_g = transposeTensor(hidden_g);
    }
  }

  /** @todo use context->getName() once context is enabled */
  Var_Grad in_var(i_dim, true, false, "dist_layer:input");
  Var_Grad out_var(h_dim, true, false, "dist_layer:output");

  for (unsigned int i = 0; i < in_dim.height(); ++i) {
    //
    // Iterate Height Direction. The dimension of in is input_[ b, 1, 1, width
    // ]. The dimension of out is hidden_[ b, 1, 1, width ]
    //
    Tensor label_iter;

    Tensor in_iter =
      in.getSharedDataTensor(i_dim, i * in_dim.batch() * in_dim.width());
    Tensor out_iter =
      out.getSharedDataTensor(h_dim, i * ho_dim.batch() * ho_dim.width());

    in_var.initializeVariable(in_iter);
    out_var.initializeVariable(out_iter);

    if (dist_layer->getType() == "loss") {
      label_iter =
        h_g.getSharedDataTensor(h_dim, i * ho_dim.batch() * ho_dim.width());
      out_var.initializeGradient(label_iter);
    }

    dist_layer->setInputBuffers({std::make_shared<Var_Grad>(in_var)});
    dist_layer->setOutputBuffers({std::make_shared<Var_Grad>(out_var)});

    dist_layer->forwarding();
  }

  hidden_.copy(transposeTensor(out));
}

void TimeDistLayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

  std::shared_ptr<TimeDistLayer> from =
    std::static_pointer_cast<TimeDistLayer>(l);
  this->dist_layer = from->dist_layer;
}

void TimeDistLayer::setDistLayer(std::shared_ptr<LayerV1> l) {
  dist_layer = l;
  LayerV1::setActivation(l->getActivationType());
};

void TimeDistLayer::calcDerivative() {
  Tensor &derivative_ = net_hidden[0]->getGradientRef();
  Tensor &hval_ = net_hidden[0]->getVariableRef();
  Tensor &ret_ = net_input[0]->getGradientRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  TensorDim der_dim = derivative_.getDim();
  TensorDim ret_dim = ret_.getDim();

  TensorDim r_dim = {ret_dim[2], 1, 1, ret_dim[3]};
  TensorDim d_dim = {der_dim[2], 1, 1, der_dim[3]};

  /** @todo use context->getName() once context is enabled */
  Var_Grad in_var(r_dim, true, false, "dist_layer:input");
  Var_Grad out_var(d_dim, true, false, "dist_layer:output");

  for (unsigned int i = 0; i < der_dim[0]; ++i) {
    Tensor ret_iter =
      ret_.getSharedDataTensor(r_dim, i * r_dim.batch() * r_dim.width());
    Tensor in_iter =
      input_.getSharedDataTensor(r_dim, i * r_dim.batch() * r_dim.width());
    Tensor d_iter =
      derivative_.getSharedDataTensor(d_dim, i * d_dim.batch() * d_dim.width());
    Tensor hval_iter =
      hval_.getSharedDataTensor(d_dim, i * d_dim.batch() * d_dim.width());

    in_var.initializeGradient(ret_iter);
    in_var.initializeVariable(in_iter);
    out_var.initializeGradient(d_iter);
    out_var.initializeVariable(hval_iter);

    dist_layer->setInputBuffers({std::make_shared<Var_Grad>(in_var)});
    dist_layer->setOutputBuffers({std::make_shared<Var_Grad>(out_var)});

    dist_layer->calcDerivative();
  }

  ret_.copy(transposeTensor(ret_));
  // We are not going to transpose the data. The Date is not used anymore.
  // It will be overwritten at next iteration
  // Just reshpae the tensors
  hval_.reshape({der_dim[2], 1, der_dim[0], der_dim[3]});
  derivative_.reshape({der_dim[2], 1, der_dim[0], der_dim[3]});
  input_.reshape({ret_dim[2], 1, ret_dim[0], ret_dim[3]});
}

void TimeDistLayer::calcGradient() {
  // Even if the dist_layer->getNumWeights() == 0, We do transpose here
  // for the calculation of derivatives and overwrite original tensors.
  // And use them in calcDerivatives() without transpose.
  transposeInOut();

  if (dist_layer->getNumWeights() == 0)
    return;

  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor &derivative_ = net_hidden[0]->getGradientRef();

  TensorDim der_dim = derivative_.getDim();
  TensorDim in_dim = input_.getDim();

  TensorDim i_dim = {in_dim[2], 1, 1, in_dim[3]};
  TensorDim d_dim = {der_dim[2], 1, 1, der_dim[3]};

  for (unsigned int i = 0; i < der_dim[0]; ++i) {
    Tensor in_iter =
      input_.getSharedDataTensor(i_dim, i * i_dim.batch() * i_dim.width());
    Tensor d_iter =
      derivative_.getSharedDataTensor(d_dim, i * d_dim.batch() * d_dim.width());

    /** @todo use context->getName() once context is enabled */
    Var_Grad in_var(i_dim, true, false, "dist_layer:input");
    Var_Grad out_var(d_dim, true, false, "dist_layer:output");

    in_var.initializeVariable(in_iter);
    out_var.initializeGradient(d_iter);

    dist_layer->setInputBuffers({std::make_shared<Var_Grad>(in_var)});
    dist_layer->setOutputBuffers({std::make_shared<Var_Grad>(out_var)});

    dist_layer->calcGradient();
  }
}

} /* namespace nntrainer */
