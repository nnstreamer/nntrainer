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

#include <layer.h>
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
    Layer(LayerType::LAYER_CONV2D, args...),
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
  int initialize();

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @copydoc Layer::forwarding(sharedConstTensor in)
   */
  sharedConstTensor forwarding(sharedConstTensor in);

  /**
   * @copydoc Layer::backwarding(sharedConstTensor in, int iteration)
   */
  sharedConstTensor backwarding(sharedConstTensor in, int iteration);

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
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Convolution2D"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

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
   * @brief     set normalization
   * @param[in] enable boolean
   */
  void setNormalization(bool enable) { this->normalization = enable; };

  /**
   * @brief     set standardization
   * @param[in] enable boolean
   */
  void setStandardization(bool enable) { this->standardization = enable; };

  /**
   * @brief     calculation convolution
   * @param[in] in input tensor data
   * @param[in] indim input tensor dimension
   * @param[in] kernel convolution kernel data
   * @param[in] kdim convolution kernel dimension
   * @param[in] out output
   * @param[in] stride stride value : x, y direction
   * @param[in] bias bias data
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int conv2d(float *in, TensorDim indim, const float *kernel, TensorDim kdim,
             float *out, unsigned int const *stride, float bias);

  /**
   * @brief     calculation convolution with cblas_*gemm
   * @param[in] mkernel kernel data
   * @param[in] kdim kernel data demension
   * @param[in] in input tensor
   * @param[in] outdim output tensor dimension
   * @param[in] stride stride value : x, y direction
   * @param[in] padd pad value : x, y direction
   * @param[out] out output data
   * @param[in] osize output size
   * @param[in] channel_mode loop with channel first
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int conv2d_gemm(const float *mkernel, TensorDim kdim, Tensor const &in,
                  TensorDim outdim,
                  const std::array<unsigned int, CONV2D_DIM> &stride,
                  const std::array<unsigned int, CONV2D_DIM> &pad, float *out,
                  unsigned int osize, bool channel_mode);

  /**
   * @brief     reform the data to 2d matrix
   * @param[in] in_padded padded input data
   * @param[in] kdim kernel dimesion for define number of row
   * @param[out] inCol reformed data
   * @param[in] outdim output dimension
   * @param[in] mstride stride value : x, y direction
   * @param[in] channel_mode loop with channel first
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int im2col(Tensor in_padded, TensorDim kdim, float *inCol, TensorDim outdim,
             const std::array<unsigned int, CONV2D_DIM> &mstride,
             bool channel_mode);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */
