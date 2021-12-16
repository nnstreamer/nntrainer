// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   identity.h
 * @date   16 Dec 2021
 * @brief  This is identity layer flows everything as it is
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __IDENTITY_LAYER_H__
#define __IDENTITY_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Identity Layer
 * @note    Identity layers takes multiple tensors as input, redirects to output
 * without doing nothing (or if unavoidable, copying)
 */
class IdentityLayer final : public Layer {
public:
  /**
   * @brief     Constructor of IdentityLayer
   */
  IdentityLayer();

  /**
   * @brief     Destructor of IdentityLayer
   */
  ~IdentityLayer();

  /**
   *  @brief  Move constructor of Identity Layer.
   *  @param rhs target
   */
  IdentityLayer(IdentityLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param rhs IdentityLayer to be moved.
   */
  IdentityLayer &operator=(IdentityLayer &&rhs) noexcept = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return IdentityLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "identity";
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __IDENTITY_LAYER_H__ */
