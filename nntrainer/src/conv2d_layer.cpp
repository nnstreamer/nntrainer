// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	conv2d_layer.h
 * @date	02 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */

#include <blas_interface.h>
#include <conv2d_layer.h>
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <string>
#include <util_func.h>

namespace nntrainer {

int Conv2DLayer::initialize() {
  int status = ML_ERROR_NONE;

  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  TensorDim dim =
    TensorDim(1, input_dim.channel(), kernel_size[0], kernel_size[1]);
  TensorDim bias_dim = TensorDim(1, 1, 1, 1);

  std::string kernelPrefix = "Conv2d:filter";
  std::string biasPrefix = "Conv2d:bias";
  setParamSize(filter_size * 2);

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor Knl = initializeWeight(dim, weight_initializer, status);
    NN_RETURN_STATUS();

    Tensor bias = initializeWeight(bias_dim, bias_initializer, status);
    NN_RETURN_STATUS();

    Tensor delK(dim);
    delK.setZero();

    Tensor delBias(bias_dim);
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

sharedConstTensor Conv2DLayer::forwarding(sharedConstTensor in) {
  int status = ML_ERROR_NONE;
  input = *in;

  if (normalization) {
    input = input.normalization();
  }

  if (standardization) {
    input = input.standardization();
  }

  TensorDim hidden_dim = output_dim;
  hidden_dim.batch(in->batch());
  hidden = Tensor(hidden_dim);
  hidden.setZero();

  TensorDim kdim(filter_size, input_dim.channel(), kernel_size[0],
                 kernel_size[1]);

  std::vector<float> imkernel(kdim.getFeatureLen() * filter_size);

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor &filters = paramsAt(i).weight;
    float *d = imkernel.data();
    memcpy(&d[i * kdim.getFeatureLen()], filters.getData(),
           kdim.getFeatureLen() * sizeof(float));
  }

  for (unsigned int b = 0; b < input.batch(); ++b) {
    std::vector<float> out(output_dim.getFeatureLen());
    Tensor inSub(TensorDim(1, input.channel(), input.height(), input.width()),
                 input.getAddress(b * input.getDim().getFeatureLen()));

    status = conv2d_gemm(imkernel.data(), kdim, inSub, output_dim, stride,
                         padding, out.data(), out.size(), true);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Forwarding Convolution failed.");
    memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen()), out.data(),
           out.size() * sizeof(float));

    for (unsigned int i = 0; i < filter_size; i++) {
      Tensor &bias = paramsAt(i + filter_size).weight;
      Tensor tmp(1, 1, hidden.height(), hidden.width());
      tmp.setValue(bias.getValue(0, 0, 0, 0));
      saxpy(hidden.height() * hidden.width(), 1, tmp.getData(), 1,
            hidden.getAddress(b * hidden.getDim().getFeatureLen() +
                              i * hidden.height() * hidden.width()),
            1);
    }
  }

  loss = 0.0f;
  if (weight_regularizer.type == WeightRegularizerType::l2norm) {
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &weight = paramsAt(i).weight;
      loss += weight_regularizer.constant * 0.5f * (weight.l2norm());
    }
    loss /= filter_size;
  }

  return MAKE_SHARED_TENSOR(hidden);
};

sharedConstTensor Conv2DLayer::backwarding(sharedConstTensor derivative,
                                           int iteration) {

  unsigned int same_pad[CONV2D_DIM];

  same_pad[0] = kernel_size[0] - 1;
  same_pad[1] = kernel_size[1] - 1;

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor &delK = paramsAt(i).grad;
    Tensor &delBias = paramsAt(i + filter_size).grad;
    delK.setZero();
    delBias.setZero();
  }

  Tensor ret(input_dim.batch(), input_dim.channel(),
             input_dim.height() + padding[0] * 2,
             input_dim.width() + padding[1] * 2);
  ret.setZero();

  int status = ML_ERROR_NONE;
  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    std::vector<float> out(kernel_size[0] * kernel_size[1] *
                           input_dim.channel() * filter_size);

    Tensor inSub(
      TensorDim(1, input_dim.channel(), input_dim.height(), input_dim.width()),
      input.getAddress(b * input.getDim().getFeatureLen()));

    status = conv2d_gemm(
      derivative->getAddress(b * derivative->getDim().getFeatureLen()),
      TensorDim(1, derivative->channel(), derivative->height(),
                derivative->width()),
      inSub,
      TensorDim(1, 1, filter_size,
                kernel_size[0] * kernel_size[1] * input_dim.channel()),
      stride, padding, out.data(), out.size(), false);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Backwarding Convolution failed.");

    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &delK = paramsAt(i).grad;
      Tensor &delBias = paramsAt(i + filter_size).grad;
      float *del = delK.getData();
      unsigned int s = kernel_size[0] * kernel_size[1] * input_dim.channel();

      for (unsigned int k = 0; k < s; ++k) {
        del[k] += out[i * s + k];
      }

      float sum = 0.0;
      for (unsigned int j = 0; j < derivative->height(); ++j) {
        for (unsigned int k = 0; k < derivative->width(); ++k) {
          sum += derivative->getValue(b, i, j, k);
        }
      }
      delBias.setValue(0, 0, 0, 0, sum + delBias.getValue(0, 0, 0, 0));
    }
  }

  // Calculate Derivative to propagation
  TensorDim kdim(ret.channel(), filter_size, kernel_size[0], kernel_size[1]);

  std::vector<float> imkernel(kdim.getDataLen());

  unsigned int count = 0;
  float *d = imkernel.data();

  for (unsigned int j = 0; j < ret.channel(); ++j) {
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &filters = paramsAt(i).weight;
      for (unsigned int k = 0; k < kernel_size[0] * kernel_size[1]; ++k) {
        d[count++] = filters.getData()[j * kernel_size[0] * kernel_size[1] + k];
      }
    }
  }

  TensorDim input_dim_padded(1, input_dim.channel(),
                             input_dim.height() + padding[0] * 2,
                             input_dim.width() + padding[1] * 2);

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor inSub(
      TensorDim(1, derivative->channel(), derivative->height(),
                derivative->width()),
      derivative->getAddress(b * derivative->getDim().getFeatureLen()));

    status =
      conv2d_gemm(imkernel.data(), kdim, inSub, input_dim_padded, stride,
                  same_pad, ret.getAddress(b * ret.getDim().getFeatureLen()),
                  input_dim_padded.getFeatureLen(), true);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Backwarding Convolution failed.");
  }

  if (trainable) {
    //  Update K / bias
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &delK = paramsAt(i).grad;
      Tensor &filters = paramsAt(i).weight;

      delK = delK.chain()
               .applyIf(this->isWeightRegularizerL2Norm(), _LIFT(add_i),
                        filters, weight_regularizer.constant)
               .run();
    }

    opt.apply_gradients(params, param_size, iteration);
  }

  return MAKE_SHARED_TENSOR(std::move(strip_pad(ret, padding)));
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
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
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
    ml_loge("Error: number of filters must be greater than 0");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  filter_size = f;
  return status;
}

void Conv2DLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::filters: {
    if (!value.empty()) {
      status = setUint(filter_size, value);
      throw_status(status);
    }
  } break;
  case PropertyType::kernel_size:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(kernel_size));
      throw_status(status);
      if (kernel_size[0] == 0 || kernel_size[1] == 0) {
        throw std::invalid_argument(
          "[Conv2DLayer] kernel_size must be greater than 0");
      }
    }
    break;
  case PropertyType::stride:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(stride));
      throw_status(status);
      if (stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument(
          "[Conv2DLayer] stride must be greater than 0");
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

int Conv2DLayer::conv2d(float *in, TensorDim indim, const float *kernel,
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
  for (unsigned int j = 0; j < height - k_height + 1; j += stride[0]) {
    J = 0;
    for (unsigned int k = 0; k < width - k_width + 1; k += stride[1]) {
      float sum = 0.0f;
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

int Conv2DLayer::conv2d_gemm(const float *mkernel, TensorDim kdim,
                             Tensor const &in, TensorDim outdim,
                             unsigned int const *mstride,
                             unsigned int const *pad, float *out,
                             unsigned int osize, bool channel_mode) {
  int status = ML_ERROR_NONE;
  std::vector<float> in_col;

  if (channel_mode) {
    in_col.resize(kdim.getFeatureLen() * outdim.width() * outdim.height());
  } else {
    in_col.resize(kdim.width() * kdim.height() * outdim.width());
  }

  Tensor in_padded = zero_pad(0, in, pad);
  status =
    im2col(in_padded, kdim, in_col.data(), outdim, mstride, channel_mode);
  if (status != ML_ERROR_NONE)
    throw std::runtime_error("Forwarding Convolution failed.");

  float alpha_dgemm = 1.0f;
  float beta_dgemm = 0.0f;
  const float *data = mkernel;
  const float *mdata = in_col.data();
  float *rdata = out;

  unsigned int kh, kw, w;

  if (channel_mode) {
    kh = kdim.batch();
    kw = kdim.getFeatureLen();
    w = outdim.width() * outdim.height();
  } else {
    kh = outdim.height();
    kw = kdim.width() * kdim.height();
    w = outdim.width();
  }

  sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kh, w, kw, alpha_dgemm, data,
        kw, mdata, w, beta_dgemm, rdata, w);

  return status;
}

int Conv2DLayer::im2col(Tensor in_padded, TensorDim kdim, float *in_col,
                        TensorDim outdim, unsigned int const *mstride,
                        bool channel_mode) {

  int status = ML_ERROR_NONE;
  unsigned int count;
  unsigned int channel = in_padded.channel();
  unsigned int height = in_padded.height();
  unsigned int width = in_padded.width();
  unsigned int k_width = kdim.width();
  unsigned int k_height = kdim.height();

  unsigned int J = 0;
  if (channel_mode) {
    for (unsigned int j = 0; j <= height - k_height; j += mstride[0]) {
      for (unsigned int k = 0; k <= width - k_width; k += mstride[1]) {
        count = 0;
        for (unsigned int i = 0; i < channel; ++i) {
          for (unsigned int ki = 0; ki < k_height; ++ki) {
            for (unsigned int kj = 0; kj < k_width; ++kj) {
              in_col[count * (outdim.width() * outdim.height()) + J] =
                in_padded
                  .getData()[i * height * width + (j + ki) * width + (k + kj)];
              count++;
            }
          }
        }
        J++;
      }
    }
    if (J != outdim.width() * outdim.height())
      status = ML_ERROR_INVALID_PARAMETER;
  } else {
    for (unsigned int i = 0; i < channel; ++i) {
      for (unsigned int j = 0; j <= height - k_height; j += mstride[0]) {
        for (unsigned int k = 0; k <= width - k_width; k += mstride[1]) {
          count = 0;
          for (unsigned int ki = 0; ki < k_height; ++ki) {
            for (unsigned int kj = 0; kj < k_width; ++kj) {
              in_col[count * (outdim.width()) + J] =
                in_padded
                  .getData()[i * height * width + (j + ki) * width + (k + kj)];
              count++;
            }
          }
          J++;
        }
      }
    }
    if (J != outdim.width())
      status = ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

} /* namespace nntrainer */
