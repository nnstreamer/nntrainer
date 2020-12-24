// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	conv2d_layer.h
 * @date	01 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */

#ifndef __CONV2D_LAYER_H_
#define __CONV2D_LAYER_H_
#ifdef __cplusplus

#include <layer_internal.h>
#include <manager.h>
#include <memory.h>
#include <tensor.h>

namespace nntrainer {

constexpr const unsigned int CONV2D_DIM = 2;

/**
 * @class   Convolution 2D Layer
 * @brief   Convolution 2D Layer
 */
class Conv2DLayer : public Layer {
public:
  /**
   * @brief     Constructor of Conv 2D Layer
   */
  template <typename... Args>
  Conv2DLayer(unsigned int filter_size_ = 0,
              const std::array<unsigned int, CONV2D_DIM> &kernel_size_ = {0, 0},
              const std::array<unsigned int, CONV2D_DIM> &stride_ = {1, 1},
              const std::array<unsigned int, CONV2D_DIM> &padding_ = {0, 0},
              bool normalization_ = false, bool standardization_ = false,
              Args... args) :
    Layer(args...),
    filter_size(filter_size_),
    kernel_size(kernel_size_),
    stride(stride_),
    padding(padding_),
    normalization(normalization_),
    standardization(standardization_) {}

  /**
   * @brief     Destructor of Conv 2D Layer
   */
  ~Conv2DLayer() {}

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] Conv2dLayer &&
   */
  Conv2DLayer(Conv2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Conv2DLayer to be moved.
   */
  Conv2DLayer &operator=(Conv2DLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager);

  /**
   * @copydoc Layer::forwarding(sharedConstTensors in)
   */
  void forwarding(sharedConstTensors in);

  /**
   * @copydoc Layer::calcDerivative(sharedConstTensors in)
   */
  void calcDerivative(sharedConstTensors in);

  /**
   * @copydoc Layer::calcGradient(sharedConstTensors in)
   */
  void calcGradient(sharedConstTensors in);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /* TO DO : support keras type of padding */
  /* enum class PaddingType { */
  /*   full = 0, */
  /*   same = 1, */
  /*   valid = 2, */
  /*   unknown = 3, */
  /* }; */

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const { return Conv2DLayer::type; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

  static const std::string type;

  /**
   * @copydoc Layer::scaleSize(float scalesize)
   */
  void scaleSize(float scalesize) noexcept;

private:
  unsigned int filter_size;
  std::array<unsigned int, CONV2D_DIM> kernel_size;
  std::array<unsigned int, CONV2D_DIM> stride;
  std::array<unsigned int, CONV2D_DIM> padding;

  bool normalization;
  bool standardization;

  /**
   * @brief     set Parameter Size
   * @param[in] * size : size arrary
   * @param[in] type : Property type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setSize(int *size, PropertyType type);

  /**
   * @brief     set Parameter Size
   * @param[in] f number of filters
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setFilter(int f);

  /**
   * @brief     calculation convolution with cblas_*gemm
   * @param[in] mkernel kernel data
   * @param[in] kdim kernel data demension
   * @param[in] in input tensor
   * @param[in] outdim output tensor dimension
   * @param[out] out output data
   * @param[in] channel_mode loop with channel first,
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int conv2d_gemm(const float *mkernel, TensorDim kdim, const float *in,
                  TensorDim outdim, float *out, bool channel_mode,
                  float beta_dgemm = 0.0f);

  /**
   * @brief     reform the data to 2d matrix
   * a region is sampled considering @a padding, @a mstride of unit @a kdim
   * Each region is mapped to one column,
   * if channel mode, kernel channel is considered part of kernel feature
   * if not, kernel channel is consider part of output dimension
   *
   * @param[in] in input data
   * @param[in] kdim kernel dimesion for define number of row
   * @param[in] padding padding information
   * @param[in] mstride stride value : x, y direction
   * @param[in] channel_mode loop with channel first
   * @return Tensor im2col tensor
   */
  Tensor im2col(const Tensor &in, const TensorDim &kdim,
                const std::array<unsigned int, CONV2D_DIM> &padding,
                const std::array<unsigned int, CONV2D_DIM> &mstride,
                bool channel_mode);

  int im2col_(Tensor in_padded, TensorDim kdim, float *in_col, TensorDim outdim,
              const std::array<unsigned int, CONV2D_DIM> &mstride,
              bool channel_mode);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */
