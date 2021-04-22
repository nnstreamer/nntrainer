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
  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  Tensor hidden_g, h_g;

  TensorDim ho_dim = hidden_.getDim();
  TensorDim in_dim = input_.getDim();

  // input_.dim = [ 1, b, h, w ] : nntrainer only support 1,2,3 transpose
  input_.reshape({1, in_dim[0], in_dim[2], in_dim[3]});
  Tensor in = input_.transpose("1:0:2");
  // now in.dim = [1, h, b, w]
  in.reshape({in_dim[2], in_dim[1], in_dim[0], in_dim[3]});
  // now in.dim = [h, 1, b, w]

  Tensor out = Tensor({ho_dim[2], 1, ho_dim[0], ho_dim[3]}, true);
  // now out.dim = [h, 1, b, w]

  TensorDim i_dim = in_dim;
  i_dim.channel(1);
  i_dim.height(1);

  TensorDim h_dim = ho_dim;
  h_dim.channel(1);
  h_dim.height(1);

  if (dist_layer->getType() == "loss") {
    hidden_g = net_hidden[0]->getGradientRef();
    if (!hidden_g.uninitialized()) {
      hidden_g.reshape({1, ho_dim[0], ho_dim[2], ho_dim[3]});

      h_g = hidden_g.transpose("1:0:2");
      h_g.reshape({ho_dim[2], 1, ho_dim[0], ho_dim[3]});
    }
  }

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

    Var_Grad in_var(i_dim, true, false, dist_layer->getName() + ":input");
    Var_Grad out_var(h_dim, true, false, dist_layer->getName() + ":output");

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

  input_.reshape(in_dim);
  out.reshape({1, ho_dim[2], ho_dim[0], ho_dim[3]});
  hidden_.copy(out.transpose("1:0:2"));

  hidden_.reshape(ho_dim);
  if (dist_layer->getType() == "loss")
    hidden_g.reshape(ho_dim);
}

void TimeDistLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<TimeDistLayer> from =
    std::static_pointer_cast<TimeDistLayer>(l);
  this->dist_layer = from->dist_layer;
}

void TimeDistLayer::setDistLayer(std::shared_ptr<Layer> l) {
  dist_layer = l;
  Layer::setActivation(l->getActivationType());
};

void TimeDistLayer::calcDerivative() {
  // NYI
}

void TimeDistLayer::calcGradient() {
  // NYI
}

} /* namespace nntrainer */
