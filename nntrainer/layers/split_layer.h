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
 * @todo   Add support for number of splits and uneven splits. For now, this can
 * be acheived with combination of split and concat layers.
 */

#ifndef __SPLIT_LAYER_H__
#define __SPLIT_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

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
  SplitLayer(unsigned int split_dim = 1) :
    Layer(),
    split_dimension(split_dim),
    leading_helper_dim(1) {}

  /**
   * @brief     Destructor of Split Layer
   */
  ~SplitLayer() = default;

  /**
   *  @brief  Move constructor of SplitLayer.
   *  @param[in] SplitLayer &&
   */
  SplitLayer(SplitLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SplitLayer to be moved.
   */
  SplitLayer &operator=(SplitLayer &&rhs) = default;

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
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SplitLayer::type; };

  inline static const std::string type = "split";

  /**
   * @copydoc Layer::setBatch(InitLayerContext &context, unsigned int batch)
   */
  void setBatch(InitLayerContext &context, unsigned int batch) override {
    setBatch(batch);
  }

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override {
    setBatch(batch);
  }

private:
  unsigned int split_dimension; /** dimension along which to split the input */
  unsigned int leading_helper_dim; /**< batch dimension of helper dimension not
                                containing the actual batch */
  TensorDim input_reshape_helper;  /** helper dimension to reshape input */
  TensorDim output_reshape_helper; /** helper dimension to reshape outputs */

  /**
   * @brief setProperty by type and value separated
   * @param[in] type property type to be passed
   * @param[in] value value to be passed
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const std::string &type, const std::string &value);

  /**
   * @brief set batch for the internal variables
   *
   * @param batch update batch size
   */
  void setBatch(unsigned int batch) {
    input_reshape_helper.batch(batch * leading_helper_dim);
    output_reshape_helper.batch(batch * leading_helper_dim);
  }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SPLIT_LAYER_H__ */
