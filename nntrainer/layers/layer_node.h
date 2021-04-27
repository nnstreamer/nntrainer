// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   graph_node.h
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the graph node interface for c++ API
 */

#ifndef __LAYER_NODE_H__
#define __LAYER_NODE_H__

#include <graph_node.h>
#include <layer.h>
#include <layer_internal.h>

namespace nntrainer {

/**
 * @class   LayerNode class
 * @brief   layer node class for the graph
 */
class LayerNode : public ml::train::Layer, public GraphNode {
public:
  /**
   * @brief     Constructor of LayerNode class
   *
   */
  LayerNode(std::shared_ptr<nntrainer::Layer> l, size_t idx) :
    layer(l),
    index(idx) {}

  /**
   * @brief     Destructor of LayerNode Class
   *
   */
  ~LayerNode() = default;

  /**
   * Support all the interface requirements by ml::train::Layer
   */

  /**
   * @brief Get the layer type
   *
   * @return const std::string type representation
   */
  const std::string getType() const { return layer->getType(); }

  /**
   * @brief     set Property of layer
   *
   * @param[in] properties values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  int setProperty(std::vector<std::string> properties) {
    return layer->setProperty(properties);
  }

  /**
   * @brief     Get name of the layer
   *
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  std::string getName() noexcept { return layer->getName(); }

  /**
   * Support all the interface requirements by GraphNode<nntrainer::Layer>
   */

  /**
   * @brief     Get underlying object
   *
   */
  std::shared_ptr<nntrainer::Layer> &getObject() { return layer; }

  /**
   * @brief     Get underlying object
   *
   */
  const std::shared_ptr<nntrainer::Layer> &getObject() const { return layer; }

  /**
   * @brief     Get index of the node
   *
   */
  size_t getIndex() { return index; }

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool getTrainable() noexcept { return layer->getTrainable(); }

#ifdef PROFILE
  int event_key;
#endif

private:
  std::shared_ptr<nntrainer::Layer>
    layer;      /**< The actual object in the graph node */
  size_t index; /**< index of each node */

  std::vector<std::string> input_layers;  /**< input layer names */
  std::vector<std::string> output_layers; /**< output layer names */
  bool flatten; /**< flatten the output of this node */
  ActivationType
    activation_type; /**< activation applied to the output of this node */
};

} // namespace nntrainer
#endif // __LAYER_NODE_H__
