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
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int Conv2DLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;

  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  this->last_layer = last;
  TensorDim Kdim;
  Kdim.channel(input_dim.channel());
  Kdim.height(kernel_size[0]);
  Kdim.width(kernel_size[1]);
  dim = Kdim;

  weights.clear();
  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor Knl = initializeWeight(Kdim, weight_ini_type, status);
    NN_RETURN_STATUS();

    delK.push_back(
      Tensor(input_dim.batch(), Kdim.channel(), Kdim.height(), Kdim.width()));
    delBias.push_back(Tensor(input_dim.batch(), 1, 1, 1));
    filters.push_back(Knl);
    weights.push_back(Knl);

    Tensor B(1, 1, 1, 1);
    if (!bias_init_zero) {
      B.apply([&](float x) { return random(); });
    }
    bias.push_back(B);
    weights.push_back(B);
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

void Conv2DLayer::read(std::ifstream &file) {
  std::for_each(filters.begin(), filters.end(),
                [&](Tensor &i) { i.read(file); });
  std::for_each(bias.begin(), bias.end(), [&](Tensor &i) { i.read(file); });
}

void Conv2DLayer::save(std::ofstream &file) {
  std::for_each(filters.begin(), filters.end(),
                [&](Tensor i) { i.save(file); });
  std::for_each(bias.begin(), bias.end(), [&](Tensor i) { i.save(file); });
}

Tensor Conv2DLayer::forwarding(Tensor in, int &status) {
  if (in.getDim() != input_dim) {
    status = ML_ERROR_INVALID_PARAMETER;
    return in;
  }

  if (normalization) {
    input = in.normalization();
  } else {
    input = in;
  }

  if (standardization) {
    input = input.standardization();
  }

  hidden = Tensor(output_dim);

  std::vector<float> output;

  unsigned int o_size = output_dim.width() * output_dim.height();
  output.resize(o_size);
  for (unsigned int b = 0; b < in.batch(); ++b) {
    Tensor in_padded = zero_pad(b, input, padding);
    for (unsigned int i = 0; i < filter_size; ++i) {
      status = conv2d(in_padded.getData(), in_padded.getDim(),
                      filters[i].getData(), filters[i].getDim(), output.data(),
                      stride, bias[i].getValue(b, 0, 0, 0));

      memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen() +
                               i * hidden.height() * hidden.width()),
             output.data(), o_size * sizeof(float));
    }
  }

  status = ML_ERROR_NONE;
  return hidden;
};

int Conv2DLayer::setOptimizer(Optimizer &opt) {
  int status = Layer::setOptimizer(opt);
  if (status != ML_ERROR_NONE)
    return status;

  std::vector<Tensor> list_d;
  for (unsigned int i = 0; i < filter_size; ++i) {
    for (unsigned int j = 0; j < 2; ++j) {
      list_d.push_back(Tensor(dim));
    }
    for (unsigned int j = 0; j < 2; ++j) {
      list_d.push_back(Tensor(1, 1, 1));
    }
  }

  return this->opt.initialize(list_d, true);
}

Tensor Conv2DLayer::backwarding(Tensor derivative, int iteration) {

  // Calculate delK : [batch, channel, height, width ] * filter_size
  std::vector<float> output;
  unsigned int same_pad[CONV2D_DIM];
  unsigned int o_size = kernel_size[0] * kernel_size[1];

  output.resize(o_size);

  TensorDim in_dim(1, 1, derivative.height(), derivative.width());
  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor in_padded = zero_pad(b, input, padding);
    TensorDim p_dim(1, 1, in_padded.height(), in_padded.width());

    for (unsigned int i = 0; i < filter_size; i++) {
      for (unsigned int j = 0; j < in_padded.channel(); ++j) {
        conv2d(
          in_padded.getAddress(j * in_padded.height() * in_padded.width()),
          p_dim,
          derivative.getAddress(b * derivative.getDim().getFeatureLen() +
                                i * derivative.height() * derivative.width()),
          in_dim, output.data(), stride, 0.0);
        memcpy(
          delK[i].getAddress(b * delK[i].getDim().getFeatureLen() + j * o_size),
          output.data(), o_size * sizeof(float));
      }

      // Calculate delBias [ batch , 1, 1, filter_size]

      float sum = 0.0;
      for (unsigned int j = 0; j < derivative.height(); ++j) {
        for (unsigned int k = 0; k < derivative.width(); ++k) {
          sum += derivative.getValue(b, i, j, k);
        }
      }
      delBias[i].setValue(0, 0, 0, 0, sum);
    }
  }

  // Calculate delS : returns ( Full pad )

  Tensor ret(input_dim.batch(), input_dim.channel(),
             input_dim.height() + padding[0] * 2,
             input_dim.width() + padding[1] * 2);

  same_pad[0] = kernel_size[0] - 1;
  same_pad[1] = kernel_size[1] - 1;

  TensorDim kdim(1, 1, kernel_size[0], kernel_size[1]);

  output.clear();
  output.resize(ret.height() * ret.width());

  for (unsigned int b = 0; b < derivative.batch(); ++b) {
    Tensor in_padded = zero_pad(b, derivative, same_pad);
    TensorDim p_dim(1, 1, in_padded.height(), in_padded.width());

    for (unsigned int in_c = 0; in_c < input_dim.channel(); ++in_c) {
      for (unsigned int i = 0; i < derivative.channel(); ++i) {

        conv2d(in_padded.getAddress(i * in_padded.height() * in_padded.width()),
               p_dim,
               filters[i].getAddress(in_c * kernel_size[0] * kernel_size[1]),
               kdim, output.data(), stride, 0.0);
        float *ret_vec = ret.getAddress(b * ret.getDim().getFeatureLen() +
                                        in_c * ret.height() * ret.width());
        for (unsigned int j = 0; j < ret.height() * ret.width(); ++j) {
          ret_vec[j] += output[j];
        }
      }
    }
  }

  gradients.clear();

  if (trainable) {
    //  Update K / bias
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor djdw = delK[i]
                      .chain()
                      .applyIf(this->isWeightDecayL2Norm(), _LIFT(add_i),
                               filters[i], weight_decay.lambda)
                      .run();

      gradients.push_back(djdw);
      gradients.push_back(delBias[i]);
    }
    opt.apply_gradients(weights, gradients, iteration);
  }

  return rotate_180(strip_pad(ret, padding));
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
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->dim = from->dim;
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
  this->last_layer = from->last_layer;
}

int Conv2DLayer::setSize(int *size, PropertyType type) {
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
    case PropertyType::filter: {
      int size;
      status = setInt(size, value);
      NN_RETURN_STATUS();
      filter_size = size;
    } break;
    case PropertyType::kernel_size:
      status = getValues(CONV2D_DIM, value, (int *)(kernel_size));
      NN_RETURN_STATUS();
      if (kernel_size[0] == 0 || kernel_size[1] == 0) {
        ml_loge("Error: kernel_size must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::stride:
      status = getValues(CONV2D_DIM, value, (int *)(stride));
      NN_RETURN_STATUS();
      if (stride[0] == 0 || stride[1] == 0) {
        ml_loge("Error: stride must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::padding:
      status = getValues(CONV2D_DIM, value, (int *)(padding));
      NN_RETURN_STATUS();
      break;
    case PropertyType::normalization:
      status = setBoolean(normalization, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::standardization:
      status = setBoolean(standardization, value);
      NN_RETURN_STATUS();
      break;
    default:
      status = Layer::setProperty({values[i]});
      NN_RETURN_STATUS();
      break;
    }
  }
  return status;
}

int Conv2DLayer::conv2d(float *in, TensorDim indim, float *kernel,
                        TensorDim kdim, float *out, unsigned int const *stride,
                        float bias) {

  int status = ML_ERROR_NONE;
  unsigned int channel = indim.channel();
  unsigned int height = indim.height();
  unsigned int width = indim.width();
  unsigned int k_width = kdim.width();
  unsigned int k_height = kdim.height();
  unsigned int o_width = ((indim.width() - kdim.width()) / stride[0] + 1);

  if (indim.channel() != kdim.channel()) {
    ml_loge("Error: Input and Kenel Dimension is not match!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  // Optimizer This routine : There are lots of dulicated calculations
  unsigned int I = 0;
  unsigned int J = 0;
  for (unsigned int j = 0; j <= height - k_height; j += stride[0]) {
    J = 0;
    for (unsigned int k = 0; k <= width - k_width; k += stride[1]) {
      float sum = 0.0;
      for (unsigned int i = 0; i < channel; ++i) {
        for (unsigned int ki = 0; ki < k_height; ++ki) {
          for (unsigned int kj = 0; kj < k_width; ++kj) {
            sum += kernel[i * k_height * k_width + ki * k_width + kj] *
                   in[i * height * width + (j + ki) * width + (k + kj)];
          }
        }
      }
      sum += bias;
      out[I * o_width + J] = sum;
      J++;
    }
    I++;
  }

  return status;
}

} /* namespace nntrainer */
