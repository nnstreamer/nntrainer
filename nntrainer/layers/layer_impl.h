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
  LayerImpl();

  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~LayerImpl() = default;

  /**
   *  @brief  Move constructor of LayerImpl Layer.
   *  @param[in] LayerImpl &&
   */
  LayerImpl(LayerImpl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LayerImpl to be moved.
   */
  LayerImpl &operator=(LayerImpl &&rhs) = default;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override;
  
  virtual std::string getProperty(const std::string &key) override {
    return "";
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  virtual void exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const override;

  template <typename Func, typename Tuple, std::size_t... ls>
  std::string for_each_imple(Func &&f, const Tuple &t,
                                        std::index_sequence<ls...>,
                                        const std::string &key) {
    std::string result = "";

    (..., (
            [&] {
              auto &&elem = std::get<ls>(t);
              if (strcmp(getPropKey(elem), key.c_str()) == 0) {
                if (!elem.empty()) {
                  result = to_string(elem);
                } else {
                  result = "empty";
                }
              }
            }(),
            0));

    return result;
  }

  template <typename Func, typename... Args>
  std::string for_each(const std::tuple<Args...> &t, Func &&f,
                                  const std::string &key) {
    return for_each_imple(std::forward<Func>(f), t,
                          std::index_sequence_for<Args...>{}, key);
  }

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
