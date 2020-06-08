/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	conv2d_layer.h
 * @date	02 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */

#include <conv2d_layer.h>
#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <random>
#include <util_func.h>

namespace nntrainer {

int Conv2DLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;

  if (input_dim.getDataLen() == 1) {
    ml_logw("Warnning: the length of previous layer dimension is one");
  }

  if (input_dim.batch() <= 0 || input_dim.height() <= 0 ||
      input_dim.width() <= 0 || input_dim.channel() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->last_layer = last;
  TensorDim Kdim;
  Kdim.channel(input_dim.channel());
  Kdim.height(kernel_size[0]);
  Kdim.width(kernel_size[1]);

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor Knl = initializeWeight(Kdim, weight_ini_type, status);
    NN_RETURN_STATUS();
    filters.push_back(Knl);

    Tensor B = Tensor(1, 1);
    if (init_zero) {
      B.setZero();
    } else {
      B = B.apply(random);
    }
    bias.push_back(B);
  }
  // this output_dim should be the same with dimension of hidden
  output_dim.batch(input_dim.batch());
  output_dim.channel(filter_size);
  output_dim.height(
    (input_dim.height() - kernel_size[0] + 2 * padding[0]) / stride[0] + 1);
  output_dim.width(
    (input_dim.width() - kernel_size[1] + 2 * padding[1]) / stride[1] + 1);

  return status;
}

int Conv2DLayer::initialize(int b, int c, int h, int w, bool last,
                            bool init_zero) {
  int status = ML_ERROR_NONE;

  this->input_dim.batch(b);
  this->input_dim.channel(c);
  this->input_dim.width(w);
  this->input_dim.height(h);

  this->init_zero = init_zero;
  status = this->initialize(last);

  return status;
}

void Conv2DLayer::read(std::ifstream &file) {
  // NYI
}

void Conv2DLayer::save(std::ofstream &file) {
  // NYI
}

Tensor Conv2DLayer::forwarding(Tensor in, int &status) {
  // NYI
  return in;
};

Tensor Conv2DLayer::forwarding(Tensor in, Tensor output, int &status) {
  return in;
}

Tensor Conv2DLayer::backwarding(Tensor in, int iteration) {
  // NYI
  return in;
}

void Conv2DLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<Conv2DLayer> from = std::static_pointer_cast<Conv2DLayer>(l);
  this->filter_size = from->filter_size;
  for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
    this->kernel_size[i] = from->kernel_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
  }

  for (int i = 0; from->filters.size(); ++i) {
    this->filters.push_back(from->filters[i]);
    this->bias.push_back(from->bias[i]);
  }
}

int Conv2DLayer::setSize(int *size, nntrainer::Conv2DLayer::PropertyType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::kernel_size:
    for (int i = 0; i < CONV2D_DIM; ++i) {
      kernel_size[i] = size[i];
    }
    break;
  case PropertyType::stride:
    for (int i = 0; i < CONV2D_DIM; ++i) {
      stride[i] = size[i];
    }
    break;
  case PropertyType::padding:
    for (int i = 0; i < CONV2D_DIM; ++i) {
      padding[i] = size[i];
    }
    break;
  default:
    ml_loge("Error: Unknown Layer Property type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

int Conv2DLayer::setFilter(int f) {
  int status = ML_ERROR_NONE;
  if (f <= 0) {
    ml_loge("Error: Filter size must be greater than 0");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  filter_size = f;
  return status;
}

int Conv2DLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int t = parseLayerProperty(key);

    switch (static_cast<PropertyType>(t)) {
    case PropertyType::input_shape:
      status = input_dim.setTensorDim(value.c_str());
      NN_RETURN_STATUS();
      break;
    case PropertyType::bias_zero:
      status = setBoolean(init_zero, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::activation:
      status = setActivation((ActiType)parseType(value, TOKEN_ACTI));
      NN_RETURN_STATUS();
      break;
    case PropertyType::weight_decay:
      weight_decay.type = (WeightDecayType)parseType(value, TOKEN_WEIGHT_DECAY);
      if (weight_decay.type == WeightDecayType::unknown) {
        ml_loge("Error: Unknown Weight Decay");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::weight_decay_lambda:
      status = setFloat(weight_decay.lambda, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::weight_ini:
      weight_ini_type = (WeightIniType)parseType(value, TOKEN_WEIGHTINI);
      break;
    case PropertyType::filter: {
      int size;
      status = setInt(size, value);
      NN_RETURN_STATUS();
      filter_size = size;
    } break;
    case PropertyType::kernel_size:
      status = getValues(CONV2D_DIM, value, (int *)(kernel_size));
      if (kernel_size[0] == 0 || kernel_size[1] == 0) {
        ml_loge("Error: stride must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      NN_RETURN_STATUS();
      break;
    case PropertyType::stride:
      status = getValues(CONV2D_DIM, value, (int *)(stride));
      if (stride[0] == 0 || stride[1] == 0) {
        ml_loge("Error: stride must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      NN_RETURN_STATUS();
      break;
    case PropertyType::padding:
      status = getValues(CONV2D_DIM, value, (int *)(padding));
      NN_RETURN_STATUS();
      break;
    default:
      ml_loge("Error: Unknown Layer Property Key : %s", key.c_str());
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

} /* namespace nntrainer */
