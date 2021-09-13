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
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ExportMethods &methods)
   */
  virtual void exportTo(Exporter &exporter,
                        const ExportMethods &method) const override;

protected:
  std::unique_ptr<
    std::tuple<props::WeightRegularizer, props::WeightRegularizerConstant,
               props::WeightInitializer, props::BiasInitializer>>
    layer_impl_props; /**< layer_impl_props */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_IMPL_H__ */
