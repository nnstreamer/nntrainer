// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.h
 * @date   27 Oct 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 */

#ifndef __CONCAT_LAYER_H__
#define __CONCAT_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Concat Layer
 * @brief   Concat Layer
 */
class ConcatLayer : public Layer {
public:
  /**
   * @brief     Constructor of Concat Layer
   */
  ConcatLayer() : Layer() {}

  /**
   * @brief     Destructor of Concat Layer
   */
  ~ConcatLayer() = default;

  /**
   *  @brief  Move constructor of ConcatLayer.
   *  @param[in] ConcatLayer &&
   */
  ConcatLayer(ConcatLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ConcatLayer to be moved.
   */
  ConcatLayer &operator=(ConcatLayer &&rhs) = default;

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
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ConcatLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "concat";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONCAT_LAYER_H__ */
