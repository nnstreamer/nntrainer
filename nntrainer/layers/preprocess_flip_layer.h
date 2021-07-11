// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   preprocess_flip_layer.h
 * @date   20 January 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Preprocess Random Flip Layer Class for Neural Network
 *
 */

#ifndef __PREPROCESS_FLIP_LAYER_H__
#define __PREPROCESS_FLIP_LAYER_H__
#ifdef __cplusplus

#include <random>

#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Preprocess FLip Layer
 * @brief   Preprocess FLip Layer
 */
class PreprocessFlipLayer : public Layer {
public:
  /**
   * @brief     Constructor of Preprocess FLip Layer
   */
  PreprocessFlipLayer() :
    Layer(),
    flipdirection(FlipDirection::horizontal_and_vertical) {}

  /**
   * @brief     Destructor of Preprocess FLip Layer
   */
  ~PreprocessFlipLayer() = default;

  /**
   *  @brief  Move constructor of PreprocessLayer.
   *  @param[in] PreprocessLayer &&
   */
  PreprocessFlipLayer(PreprocessFlipLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PreprocessLayer to be moved.
   */
  PreprocessFlipLayer &operator=(PreprocessFlipLayer &&rhs) = default;

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
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return PreprocessFlipLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "preprocess_flip";

private:
  /** String names for the flip direction */
  static const std::string flip_horizontal;
  static const std::string flip_vertical;
  static const std::string flip_horizontal_vertical;

  /**
   * @brief Direction of the flip for data
   */
  enum class FlipDirection { horizontal, vertical, horizontal_and_vertical };

  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    flip_dist;                 /**< uniform random distribution */
  FlipDirection flipdirection; /**< direction of flip */

  /**
   * @brief setProperty by type and value separated
   * @param[in] type property type to be passed
   * @param[in] value value to be passed
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const std::string &type, const std::string &value);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __PREPROCESS_FLIP_LAYER_H__ */
