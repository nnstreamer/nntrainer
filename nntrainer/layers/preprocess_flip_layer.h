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

#include <common_properties.h>
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
  NNTR_API PreprocessFlipLayer();

  /**
   * @brief     Destructor of Preprocess FLip Layer
   */
  NNTR_API ~PreprocessFlipLayer() = default;

  /**
   *  @brief  Move constructor of PreprocessLayer.
   *  @param[in] PreprocessLayer &&
   */
  NNTR_API PreprocessFlipLayer(PreprocessFlipLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PreprocessLayer to be moved.
   */
  NNTR_API PreprocessFlipLayer &operator=(PreprocessFlipLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_API void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_API bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const override {
    return PreprocessFlipLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "preprocess_flip";

private:
  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    flip_dist; /**< uniform random distribution */
  std::tuple<props::FlipDirection> preprocess_flip_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __PREPROCESS_FLIP_LAYER_H__ */
