// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_impl.h
 * @date   21 June 2021
 * @brief  This is base Optimizer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details LayerImpl forms the base class for all the layer with weights and
 * bias parameters. LayerImpl provides parsing of properties like Weight/bias
 * initializer and regularizers. LayerImpl also provides checks for double calls
 * to finalize function.
 *
 */
#ifndef __LAYER_IMPL_H__
#define __LAYER_IMPL_H__
#ifdef __cplusplus

#include <layer_devel.h>
#include <weight.h>

#include <memory>
#include <tuple>

namespace nntrainer {

class InitLayerContext;
class RunLayerContext;
class Exporter;

enum class ExportMethods;

/**
 * @class   An abstract class to ease developing a layer
 * @brief   An abstract class for all layers
 *
 */
class LayerImpl : public Layer {

public:
  /**
   * @brief     Constructor of Layer Class
   */
  LayerImpl();

  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~LayerImpl() = default;

  /**
   * @brief     finalize the layer
   * @throw     nntrainer::not_supported if try to initialize twice
   * @copydoc   Layer::finalize(InitLayerContext &context)
   */
  virtual void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ExportMethods &methods)
   */
  virtual void exportTo(Exporter &exporter,
                        const ExportMethods &method) const override;

private:
  /**
   * @brief setProperty by type and value separated
   * @param[in] type property type to be passed
   * @param[in] value value to be passed
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  virtual void setProperty(const std::string &type, const std::string &value);

  bool finalized;                                 /**< check if finalized */
  std::unique_ptr<std::tuple<>> layer_impl_props; /**< layer_impl_props */

  WeightRegularizer weight_regularizer; /**< weight regularizer */
  float weight_regularizer_constant;    /**< weight regularizer constant */
  WeightInitializer weight_initializer; /**< initializer for the weights */
  WeightInitializer bias_initializer;   /**< initializer for the bias */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_IMPL_H__ */
