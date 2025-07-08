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

#if defined(_WIN32)
#define NNTR_API __declspec(dllexport)
#else
#define NNTR_API
#endif

namespace nntrainer {

class InitLayerContext;
class RunLayerContext;
class Exporter;

namespace props {
class WeightRegularizer;
class WeightRegularizerConstant;
class WeightInitializer;
class WeightDecay;
class BiasDecay;
class BiasInitializer;
class DisableBias;
class Print;
} // namespace props

/**
 * @class   An abstract class to ease developing a layer
 * @brief   An abstract class for all layers
 *
 */
class LayerImpl : public virtual Layer {

public:
  /**
   * @brief     Constructor of Layer Class
   */
  NNTR_API LayerImpl();

  /**
   * @brief     Destructor of Layer Class
   */
  NNTR_API virtual ~LayerImpl() = default;

  /**
   *  @brief  Move constructor of LayerImpl Layer.
   *  @param[in] LayerImpl &&
   */
  NNTR_API LayerImpl(LayerImpl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LayerImpl to be moved.
   */
  NNTR_API LayerImpl &operator=(LayerImpl &&rhs) = default;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API virtual void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getProperty(const std::string &key)
   */
  NNTR_API virtual std::string getProperty(const std::string &key) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  NNTR_API virtual void exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const override;

protected:
  std::unique_ptr<
    std::tuple<props::WeightRegularizer, props::WeightRegularizerConstant,
               props::WeightInitializer, props::WeightDecay, props::BiasDecay,
               props::BiasInitializer, props::DisableBias, props::Print>>
    layer_impl_props; /**< layer_impl_props */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_IMPL_H__ */
