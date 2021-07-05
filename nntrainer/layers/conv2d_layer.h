// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   01 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
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
class Conv2DLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Conv 2D Layer
   */
  template <typename... Args>
  Conv2DLayer(unsigned int filter_size_ = 0,
              const std::array<unsigned int, CONV2D_DIM> &kernel_size_ = {0, 0},
              const std::array<unsigned int, CONV2D_DIM> &stride_ = {1, 1},
              const std::array<unsigned int, 4> &padding_ = {0, 0, 0, 0},
              bool normalization_ = false, bool standardization_ = false,
              Args... args) :
    LayerV1(args...),
    filter_size(filter_size_),
    kernel_size(kernel_size_),
    stride(stride_),
    padding(padding_),
    conv_props(),
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
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::calcGradient()
   */
  void calcGradient() override;

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<LayerV1> l) override;

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
  const std::string getType() const override { return Conv2DLayer::type; };

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  inline static const std::string type = "conv2d";

  /**
   * @copydoc Layer::scaleSize(float scalesize)
   */
  void scaleSize(float scalesize) noexcept override;

private:
  unsigned int filter_size;
  std::array<unsigned int, CONV2D_DIM> kernel_size;
  std::array<unsigned int, CONV2D_DIM> stride;
  std::array<unsigned int, CONV2D_DIM * 2> padding;
  std::tuple<props::Padding2D> conv_props;

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
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */
