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

#include <iostream>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int Optimizer::setType(OptType t) {
  int status = ML_ERROR_NONE;
  if (t == OptType::unknown) {
    ml_loge("Error: Optimizer is unknown");
    return ML_ERROR_INVALID_PARAMETER;
  }
  type = t;
  return status;
}

int Optimizer::setOptParam(OptParam p) {
  int status = ML_ERROR_NONE;
  if (p.learning_rate <= 0) {
    ml_loge("Error: learning_rate should be grater than 0 (%f)",
            p.learning_rate);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (p.decay_steps == -1 && p.beta1 && p.beta2 && p.epsilon) {
    ml_logw("Although you set the learning rate decay param, you didn't "
            "set decay_steps");
  }

  popt = p;
  return status;
}

int Optimizer::initialize(TensorDim d, bool set_tensor) {
  int status = ML_ERROR_NONE;
  if (d.height() == 0 || d.width() == 0 || d.channel() == 0) {
    ml_loge("Error: Tensor Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (type == OptType::adam && set_tensor) {
    wm = Tensor(d.channel(), d.height(), d.width());
    wv = Tensor(d.channel(), d.height(), d.width());
    wm.setZero();
    wv.setZero();
    bm = Tensor(1, 1, d.width());
    bv = Tensor(1, 1, d.width());
    bm.setZero();
    bv.setZero();
  }
  return status;
}

void Optimizer::calculate(const Tensor &djdw, const Tensor &djdb,
                          Tensor &weight, Tensor &bias, int iteration,
                          bool init_zero) {
  Tensor djdwAvg, djdbAvg;
  float ll = popt.learning_rate;
  if (popt.decay_steps != -1) {
    ll = ll * pow(popt.decay_rate, (iteration / popt.decay_steps));
  }

  djdwAvg = djdw.average();
  djdbAvg = djdb.average();

  switch (type) {
  case OptType::sgd:
    weight.add_i(djdwAvg, -ll);
    break;
  case OptType::adam: {
    std::function<float(float)> sqrtEps = [&](float f) {
      return sqrtFloat(f) + this->popt.epsilon;
    };
    std::function<float(float)> biasCorrection = [&](float f) {
      return 1 - pow(f, iteration + 1);
    };

    ll *= sqrt(biasCorrection(popt.beta2)) / biasCorrection(popt.beta1);

    wm.multiply_i(popt.beta1);
    wm.add_i(djdwAvg, 1 - popt.beta1);

    wv.multiply_i(popt.beta2);
    wv.add_i(djdwAvg.multiply(djdwAvg), 1 - popt.beta2);

    weight.add_i(wm.divide(wv.apply(sqrtEps)), -ll);

    bm.multiply_i(popt.beta1);
    bm.add_i(djdbAvg, 1 - popt.beta1);

    bv.multiply_i(popt.beta2);
    bv.add_i(djdbAvg.multiply(djdbAvg), 1 - popt.beta2);

    bias.add_i(bm.divide(bv.apply(sqrtEps)), -ll);
    break;
  }
  case OptType::unknown:
  default:
    break;
  }

  if (init_zero) {
    bias.add_i(djdbAvg, -ll);
  }
}

int Optimizer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    status = getKeyValue(values[i], key, value);

    unsigned int type = parseOptProperty(key.c_str());

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::learning_rate:
      status = setFloat(popt.learning_rate, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::decay_steps:
      status = setFloat(popt.decay_steps, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::decay_rate:
      status = setFloat(popt.decay_rate, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::beta1:
      status = setDouble(popt.beta1, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::beta2:
      status = setDouble(popt.beta2, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::epsilon:
      status = setDouble(popt.epsilon, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::continue_train:
      status = setBoolean(popt.continue_train, value);
      NN_RETURN_STATUS();
      break;
    default:
      ml_loge("Error: Unknown Optimizer Property Key");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  return status;
}

void Optimizer::read(std::ifstream &file) {
  OptType loaded_type;
  file.read((char *)&loaded_type, sizeof(OptType));
  if (type == OptType::adam and loaded_type == type) {
    if (popt.continue_train) {
      wm.read(file);
      bm.read(file);
      wv.read(file);
      bv.read(file);
    } else {
      size_t total_size =
        wm.getSize() + bm.getSize() + wv.getSize() + bv.getSize();
      file.seekg(total_size, std::ifstream::cur);
    }
  }
}

void Optimizer::save(std::ofstream &file) {
  file.write((char *)&type, sizeof(OptType));
  if (type == OptType::adam) {
    wm.save(file);
    bm.save(file);
    wv.save(file);
    bv.save(file);
  }
}
} // namespace nntrainer
