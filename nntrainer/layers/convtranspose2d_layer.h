// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   convtranspose2d_layer.h
 * @date   01 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Transposed Convolution Layer Class for Neural Network
 *
 */

#ifndef __CONVTRANSPOSE2D_LAYER_H_
#define __CONVTRANSPOSE2D_LAYER_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

constexpr const unsigned int CONVTRANSPOSE2D_DIM = 2;

/**
 * @class   Transposed Convolution 2D Layer
 * @brief   Transposed Convolution 2D Layer
 */
class ConvTranspose2DLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Conv 2D Layer
   */
  ConvTranspose2DLayer(const std::array<unsigned int, CONVTRANSPOSE2D_DIM * 2> &padding_ = {
                0, 0, 0, 0});

  /**
   * @brief     Destructor of Conv 2D Layer
   */
  ~ConvTranspose2DLayer() = default;

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] ConvTranspose2dLayer &&
   */
  ConvTranspose2DLayer(ConvTranspose2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ConvTranspose2DLayer to be moved.
   */
  ConvTranspose2DLayer &operator=(ConvTranspose2DLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ConvTranspose2DLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /* TO DO : support keras type of padding */
  /* enum class PaddingType { */
  /*   full = 0, */
  /*   same = 1, */
  /*   valid = 2, */
  /*   unknown = 3, */
  /* }; */

  inline static const std::string type = "convtranspose2d";

private:
  std::array<unsigned int, CONVTRANSPOSE2D_DIM * 2> padding;
  std::tuple<props::FilterSize, std::array<props::KernelSize, CONVTRANSPOSE2D_DIM>,
             std::array<props::Stride, CONVTRANSPOSE2D_DIM>, props::Padding2D,
             std::array<props::Dilation, CONVTRANSPOSE2D_DIM>>
    conv_props;

  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONVTRANSPOSE2D_LAYER_H__ */
