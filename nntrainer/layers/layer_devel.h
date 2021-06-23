/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	layer_devel.h
 * @date	10 June 2021
 * @brief	This is Layer classes of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYER_DEVEL_H__
#define __LAYER_DEVEL_H__
#ifdef __cplusplus

#include <memory>
#include <string>
#include <vector>

#include <tensor.h>
#include <layer_context.h>

namespace ml::train {
class Layer;
}

namespace nntrainer {

class Exporter;

enum class ExportMethods;

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 *
 * @details nntrainer::Layer inherits ml::train::Layer but has been omitted to
 * disallow static_cast between nntrainer::Layer and ml::train::Layer objects.
 */
class Layer {

public:
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer() = default;

  /**
   * @brief Get the layer type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief     Finalize creating the layer
   * @param     context Context of the layer
   *
   * @details   Input dimensions will be provided set in the context. This
   * function must set output dimensions in the given context. Further, context
   * can be used to request weights for the layer, and any extra tensor required
   * for the operation of the layer.
   * @note      After calling this it is not allowed to
   * change properties.
   * @note      No memory allocation must be performed in the initialization
   * step. Any tensor memory required must be requested to the context which
   * will be made available during execution of the layer with the context.
   */
  virtual void finalize(InitLayerContext &context) = 0;

  /**
   * @brief     Forward Propagation of a layer
   * @param     context Context of the layer
   * @param     training true if training, false if inference
   *
   * @note      Output must be set in the output tensors.
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  virtual void forwarding(RunLayerContext &context, bool training) = 0;

  /**
   * @brief     calc the derivative to be passed to the previous layer
   * @param     context Context of the layer
   * @note      Return derivatives must be set in input gradient tensors.
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  virtual void calcDerivative(RunLayerContext &context) = 0;

  /**
   * @brief     Calculate the derivative of a layer
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   * @note      Gradients must be set in weight gradient tensors.
   */
  virtual void calcGradient(RunLayerContext &context) {}

  /**
   * @brief     set Property of layer
   * @param     values values of property
   * @throw std::invalid_argument invalid parameter.
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief this function helps exporting the layer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param     exporter exporter that conatins exporting logic
   * @param     method enum value to identify how it should be exported to
   */
  virtual void exportTo(Exporter &exporter, const ExportMethods &method) const {
  }

  /**
   * @brief Set the batch for the layer
   * @param     context Context of the layer
   * @param     batch Batch value to be set
   * @details Update the initialize context based on the updated batch size if
   * required
   */
  virtual void setBatch(InitLayerContext &context, unsigned int batch) {}

  /**
   * @brief Set the batch for the layer
   * @param     context Context of the layer
   * @param     batch Batch value to be set
   * @details Update the run context based on the updated batch size if required
   */
  virtual void setBatch(RunLayerContext &context, unsigned int batch) {}

  /**
   * @brief   If the current layer can support in-place
   *
   * @return  true if inplace, else false
   * @details all layers default to out of place execution
   * @note all layers default to out of place execution
   */
  virtual bool supportInPlace() const { return false; }

  /**
   * @brief  check if this layer requires label to be passed
   * @note   if requireLabel() == true means, for now, that it is endpoint of a
   * graph(numOutlayers == 0). label will be fed to the gradient of hidden if
   * requireLabel is true
   * @return true if requires a label when training, else false
   */
  virtual bool requireLabel() const { return false; }

  /**
   * @brief  check if this layer supports backwarding
   * @note   support backwarding primarily means that the layer can process the
   * derivatives and return back the gradients to the previous layer.
   * @return true if supports backwarding, else false
   */
  virtual bool supportBackwarding() const = 0;
};

/// @todo Decide where to put and how to implement(#986)
// /**
//  * @brief   Overriding output stream for layers and it's derived class
//  */
// template <typename T, typename std::enable_if_t<
//                         std::is_base_of<Layer, T>::value, T> * = nullptr>
// std::ostream &operator<<(std::ostream &out, T &l) {
//   // l.printPreset(out, Layer::PrintPreset::PRINT_SUMMARY);
//   return out;
// }

using CreateLayerFunc = nntrainer::Layer *(*)();
using DestroyLayerFunc = void (*)(nntrainer::Layer *);

/**
 * @brief General Layer Factory function to register Layer
 *
 * @param props property representation
 * @return std::unique_ptr<nntrainer::Layer> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<Layer, T>::value, T> * = nullptr>
std::unique_ptr<Layer> createLayer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<Layer> ptr = std::make_unique<T>();
  ptr->setProperty(props);
  return ptr;
}

/**
 * @brief  Layer Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateLayerFunc createfunc;   /**< create layer function */
  DestroyLayerFunc destroyfunc; /**< destory function */
} LayerPluggable;

/**
 * @brief pluggable layer must have this structure defined
 */
extern "C" LayerPluggable ml_train_layer_pluggable;

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_DEVEL_H__ */
