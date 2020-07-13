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
#include <string>
#include <util_func.h>

namespace nntrainer {

int Conv2DLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;

  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  this->last_layer = last;
  dim = TensorDim(1, input_dim.channel(), kernel_size[0], kernel_size[1]);

  std::string kernelPrefix = "Conv2d:filter";
  std::string biasPrefix = "Conv2d:bias";
  setParamSize(filter_size * 2);

  // fixme: #280
  TensorDim gradDim(input_dim.batch(), dim.channel(), dim.height(),
                    dim.width());
  TensorDim gradBiasDim(input_dim.batch(), 1, 1, 1);

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor Knl = initializeWeight(dim, weight_ini_type, status);
    NN_RETURN_STATUS();

    Tensor bias = Tensor(1, 1);

    if (!bias_init_zero) {
      bias.apply([&](float x) { return random(); });
    } else {
      bias.setZero();
    }
    Tensor delK(gradDim);
    delK.setZero();

    Tensor delBias(gradBiasDim);
    delBias.setZero();

    /*< @note: order of weight and bias are:
               w0 w1 w2 ... w3
    */
    paramsAt(i) = {std::move(Knl), std::move(delK),
                   kernelPrefix + std::to_string(i)};
    paramsAt(i + filter_size) = {std::move(bias), std::move(delBias),
                                 biasPrefix + std::to_string(i)};
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

void Conv2DLayer::read(std::ifstream &file) { Layer::read(file); }

void Conv2DLayer::save(std::ofstream &file) { Layer::save(file); }

Tensor Conv2DLayer::forwarding(Tensor in, int &status) {
  if (normalization) {
    input = in.normalization();
  } else {
    input = in;
  }

  if (standardization) {
    input = input.standardization();
  }

  hidden = Tensor(in.batch(), output_dim.channel(), output_dim.height(),
                  output_dim.width());
  hidden.setZero();

  std::vector<float> output(output_dim.width() * output_dim.height());

  for (unsigned int b = 0; b < in.batch(); ++b) {
    Tensor in_padded = zero_pad(b, input, padding);

    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &filter = paramsAt(i).weight;
      Tensor &bias = paramsAt(i + filter_size).weight;
      status = conv2d(in_padded.getData(), in_padded.getDim(), filter.getData(),
                      filter.getDim(), output.data(), stride,
                      bias.getValue(0, 0, 0, 0));

      memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen() +
                               i * hidden.height() * hidden.width()),
             output.data(), output.size() * sizeof(float));
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
  std::vector<Tensor>::iterator iter;
  for (iter = list_d.begin(); iter != list_d.end(); ++iter) {
    (*iter).setZero();
  }
  return this->opt.initialize(list_d, true);
}

Tensor Conv2DLayer::backwarding(Tensor derivative, int iteration) {

  // Calculate delK : [batch, channel, height, width ] * filter_size
  unsigned int same_pad[CONV2D_DIM];
  unsigned int o_size = kernel_size[0] * kernel_size[1];
  std::vector<float> output(o_size);

  TensorDim in_dim(1, 1, derivative.height(), derivative.width());
  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor in_padded = zero_pad(b, input, padding);
    TensorDim p_dim(1, 1, in_padded.height(), in_padded.width());

    /// @fixme: #280
    for (unsigned int i = 0; i < filter_size; i++) {
      Tensor &delK = paramsAt(i).grad;
      Tensor &delBias = paramsAt(i + filter_size).grad;
      for (unsigned int j = 0; j < in_padded.channel(); ++j) {
        conv2d(
          in_padded.getAddress(j * in_padded.height() * in_padded.width()),
          p_dim,
          derivative.getAddress(b * derivative.getDim().getFeatureLen() +
                                i * derivative.height() * derivative.width()),
          in_dim, output.data(), stride, 0.0);
        memcpy(delK.getAddress(b * delK.getDim().getFeatureLen() + j * o_size),
               output.data(), o_size * sizeof(float));
      }

      // Calculate delBias [ batch , 1, 1, filter_size]

      float sum = 0.0;
      for (unsigned int j = 0; j < derivative.height(); ++j) {
        for (unsigned int k = 0; k < derivative.width(); ++k) {
          sum += derivative.getValue(b, i, j, k);
        }
      }
      delBias.setValue(b, 0, 0, 0, sum);
    }
  }

  // Calculate delS : returns ( Full pad )

  Tensor ret(input_dim.batch(), input_dim.channel(),
             input_dim.height() + padding[0] * 2,
             input_dim.width() + padding[1] * 2);
  ret.setZero();

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
        Tensor &filter = paramsAt(i).weight;

        conv2d(in_padded.getAddress(i * in_padded.height() * in_padded.width()),
               p_dim, filter.getAddress(in_c * kernel_size[0] * kernel_size[1]),
               kdim, output.data(), stride, 0.0);
        float *ret_vec = ret.getAddress(b * ret.getDim().getFeatureLen() +
                                        in_c * ret.height() * ret.width());
        for (unsigned int j = 0; j < ret.height() * ret.width(); ++j) {
          ret_vec[j] += output[j];
        }
      }
    }
  }

  if (trainable) {
    //  Update K / bias
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &delK = paramsAt(i).grad;
      Tensor &filter = paramsAt(i).weight;

      delK = delK.chain()
               .applyIf(this->isWeightDecayL2Norm(), _LIFT(add_i), filter,
                        weight_decay.lambda)
               .run();
    }

    opt.apply_gradients(params, param_size, iteration);
  }

  return rotate_180(strip_pad(ret, padding));
}

void Conv2DLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);
  std::shared_ptr<Conv2DLayer> from = std::static_pointer_cast<Conv2DLayer>(l);
  this->filter_size = from->filter_size;
  for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
    this->kernel_size[i] = from->kernel_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
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

void Conv2DLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::filter: {
    if (!value.empty()) {
      int size;
      status = setInt(size, value);
      throw_status(status);
      filter_size = size;
    }
  } break;
  case PropertyType::kernel_size:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(kernel_size));
      throw_status(status);
      if (kernel_size[0] == 0 || kernel_size[1] == 0) {
        throw std::out_of_range(
          "[Conv2DLayer] kernel_size must be greater than 0");
      }
    }
    break;
  case PropertyType::stride:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(stride));
      throw_status(status);
      if (stride[0] == 0 || stride[1] == 0) {
        throw std::out_of_range("[Conv2DLayer] stride must be greater than 0");
      }
    }
    break;
  case PropertyType::padding:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(padding));
      throw_status(status);
    }
    break;
  case PropertyType::normalization:
    if (!value.empty()) {
      status = setBoolean(normalization, value);
      throw_status(status);
    }
    break;
  case PropertyType::standardization:
    if (!value.empty()) {
      status = setBoolean(standardization, value);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
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
