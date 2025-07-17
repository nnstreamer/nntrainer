// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   flatten_layer.h
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Flatten Layer Class for Neural Network
 *
 */

#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__
#ifdef __cplusplus

#include <reshape_layer.h>

namespace nntrainer {

/**
 * @class   Flatten Layer
 * @brief   Flatten Layer
 */
class FlattenLayer : public ReshapeLayer {
public:
  /**
   * @brief     Constructor of Flatten Layer
   */
  NNTR_API FlattenLayer() :
    ReshapeLayer(),
    flatten_props(props::StartDimension(), props::EndDimension()) {}

  /**
   * @brief     Destructor of Flatten Layer
   */
  NNTR_API ~FlattenLayer() = default;

  /**
   *  @brief  Move constructor of FlattenLayer.
   *  @param[in] FlattenLayer &&
   */
  NNTR_API FlattenLayer(FlattenLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FlattenLayer to be moved.
   */
  NNTR_API FlattenLayer &operator=(FlattenLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

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
    return FlattenLayer::type;
  };

  static constexpr const char *type = "flatten";

  std::tuple<props::StartDimension, props::EndDimension> flatten_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLATTEN_LAYER_H__ */
