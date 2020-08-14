/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <fc_layer.h>
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int FullyConnectedLayer::initialize(bool last, TensorParam input) {
  int status = ML_ERROR_NONE;

  this->last_layer = last;

  output_dim = input_dim;
  output_dim.width(unit);

  Tensor out(output_dim);
  TensorParam output = TensorParam(out, Tensor(), "FC:output");

  Tensor bias = Tensor(1, unit);
  if (bias_init_zero) {
    bias.setZero();
  } else {
    bias.setRandUniform(-0.5, 0.5);
  }

  TensorDim dim = output_dim;
  dim.height(input_dim.width());
  dim.batch(1);
  Tensor weight = initializeWeight(dim, weight_ini_type, status);
  NN_RETURN_STATUS();

  setParamSize(2);
  paramsAt(0) = {std::move(weight), Tensor(weight.getDim()), "FC:weight"};
  paramsAt(1) = {std::move(bias), Tensor(bias.getDim()), "FC:bias"};

  TensorParam w =
    TensorParam(paramsAt(0).weight, Tensor(weight.getDim()), "FC:weight", true);
  TensorParam b =
    TensorParam(paramsAt(1).weight, Tensor(bias.getDim()), "FC:bias", true);

  setInputTensors({input, w, b});
  setOutputTensors({output});

  return status;
}

void FullyConnectedLayer::computeOp(void) {
  Tensor &input = inputs_op[0].var;
  Tensor &weight = inputs_op[1].var;
  Tensor &bias = inputs_op[2].var;

  Tensor hidden = input.chain().dot(weight).add_i(bias).run();
  outputs_op[0].var.setData(hidden.getData());

  if (weight_decay.type == WeightDecayType::l2norm) {
    loss = weight_decay.lambda * 0.5f * (weight.l2norm());
  }
}

void FullyConnectedLayer::computeGrad(void) {
  Tensor &weight = inputs_op[1].var;
  Tensor &djdw = inputs_op[1].grad;
  Tensor &djdb = inputs_op[2].grad;
  Tensor &ret = inputs_op[0].grad;
  Tensor &derivative = outputs_op[0].grad;

  ret = derivative.dot(weight.transpose("0:2:1"));
  djdb = derivative.sum(0);
  djdw = input.chain()
           .transpose("0:2:1")
           .dot(derivative)
           .applyIf(this->isWeightDecayL2Norm(), _LIFT(add_i), weight,
                    weight_decay.lambda)
           .run()
           .sum(0);
}

void FullyConnectedLayer::applyGrad(int iteration) {
  int idx = 0;
  for (auto &tp : inputs_op) {
    if (tp.trainable && trainable) {
      opt.apply_gradients(
        std::make_shared<UpdatableParam>(params.get()[idx - 1]), 1, iteration);
    }
  }
}
} /* namespace nntrainer */
