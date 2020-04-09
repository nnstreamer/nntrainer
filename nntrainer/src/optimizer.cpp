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
 * @file	optimizer.cpp
 * @date	08 April 2020
 * @brief	This is Implementation of Optimizer class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "optimizer.h"
#include <nntrainer_log.h>
#include "util_func.h"

void Optimizer::initialize(unsigned int height, unsigned int width, bool setTensor) {
  if (type == OptType::adam && setTensor) {
    WM = Tensors::Tensor(height, width);
    WV = Tensors::Tensor(height, width);
    WM.setZero();
    WV.setZero();
    BM = Tensors::Tensor(1, width);
    BV = Tensors::Tensor(1, width);
    BM.setZero();
    BV.setZero();
  }
}

void Optimizer::calculate(Tensors::Tensor& dJdW, Tensors::Tensor& dJdB, Tensors::Tensor& Weight, Tensors::Tensor& Bias,
                          int iteration, bool init_zero) {
  if (popt.weight_decay.type == WeightDecayType::l2norm) {
    dJdW = dJdW.add(Weight.multiply(popt.weight_decay.lambda));
  }

  float ll = popt.learning_rate;
  if (popt.decay_steps != -1) {
    ll = ll * pow(popt.decay_rate, (iteration / popt.decay_steps));
  }

  switch (type) {
    case OptType::sgd:
      Weight = Weight.subtract(dJdW.average().multiply(ll));
      break;
    case OptType::adam:
      WM = WM.multiply(popt.beta1).add(dJdW.average().multiply(1 - popt.beta1));
      WV = WV.multiply(popt.beta2).add((dJdW.average().multiply(dJdW.average())).multiply(1 - popt.beta2));
      WM.divide(1 - pow(popt.beta1, iteration + 1));
      WV.divide(1 - pow(popt.beta2, iteration + 1));
      Weight = Weight.subtract((WM.divide(WV.apply(sqrt_float).add(popt.epsilon))).multiply(ll));
      BM = BM.multiply(popt.beta1).add(dJdB.average().multiply(1 - popt.beta1));
      BV = BV.multiply(popt.beta2).add((dJdB.average().multiply(dJdB.average())).multiply(1 - popt.beta2));
      BM.divide(1 - pow(popt.beta1, iteration + 1));
      BV.divide(1 - pow(popt.beta2, iteration + 1));
      Bias = Bias.subtract((BM.divide(BV.apply(sqrt_float).add(popt.epsilon))).multiply(ll));
      break;
    default:
      break;
  }

  if (init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(ll));
  }
}
