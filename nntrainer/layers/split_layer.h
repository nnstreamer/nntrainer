// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   split_layer.h
 * @date   21 May 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Split Layer Class for Neural Network
 *
 * @todo   Add support for uneven splits. For now, this can
 * be acheived with combination of split and concat layers.
 */

#ifndef __SPLIT_LAYER_H__
#define __SPLIT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <tensor_dim.h>

namespace nntrainer {

/**
 * @class   Split Layer
 * @brief   Split Layer
 */
class SplitLayer : public Layer {
public:
  /**
   * @brief     Constructor of Split Layer
   */
  NNTR_EXPORT SplitLayer();

  /**
   * @brief     Destructor of Split Layer
   */
  NNTR_EXPORT ~SplitLayer() = default;

  /**
   *  @brief  Move constructor of SplitLayer.
   *  @param[in] SplitLayer &&
   */
  NNTR_EXPORT SplitLayer(SplitLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SplitLayer to be moved.
   */
  NNTR_EXPORT SplitLayer &operator=(SplitLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_EXPORT void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_EXPORT void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const override {
    return SplitLayer::type;
  };

  static constexpr const char *type = "split";

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  NNTR_EXPORT void setBatch(RunLayerContext &context,
                         unsigned int batch) override {
    setBatch(batch);
  }

private:
  unsigned int leading_helper_dim; /**< batch dimension of helper dimension not
                                containing the actual batch */
  TensorDim input_reshape_helper;  /** helper dimension to reshape input */
  TensorDim output_reshape_helper; /** helper dimension to reshape outputs */
  std::tuple<props::SplitDimension, props::SplitNumber> split_props;

  /**
   * @brief set batch for the internal variables
   *
   * @param batch update batch size
   */
  NNTR_EXPORT void setBatch(unsigned int batch) {
    input_reshape_helper.batch(batch * leading_helper_dim);
    output_reshape_helper.batch(batch * leading_helper_dim);
  }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SPLIT_LAYER_H__ */
