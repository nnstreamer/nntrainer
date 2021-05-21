// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_node.h
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer node for network graph
 *
 * @todo   Add printPreset support
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
   * @brief     Default constructor
   */
  LayerNode() : LayerNode(nullptr) {}

  /**
   * @brief     Constructor of LayerNode class
   *
   */
  LayerNode(std::shared_ptr<nntrainer::Layer> l, size_t idx = 0);

  /**
   * @brief     Destructor of LayerNode Class
   *
   */
  ~LayerNode() = default;

  /**
   * @brief     Set the index for the node
   */
  void setIndex(size_t idx) { index = idx; }

  /**
   * Support all the interface requirements by ml::train::Layer
   */

  /**
   * @brief Get the layer type
   *
   * @return const std::string type representation
   */
  const std::string getType() const;

  /**
   * @brief     set Property of layer
   *
   * @param[in] properties values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name=property_val, ...}
   */
  int setProperty(std::vector<std::string> properties);

  /**
   * @brief     set name of layer
   *
   * @param[in] name Name of the layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setName(const std::string &name) { return setProperty({"name=" + name}); }

  /**
   * @brief     Get name of the layer
   *
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  const std::string getName() const noexcept { return getLayer()->getName(); }

  /**
   * @brief     Get name of the layer
   *
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  std::string getName() noexcept { return getLayer()->getName(); }

  /**
   * Support all the interface requirements by GraphNode<nntrainer::Layer>
   */

  /**
   * @brief     Get underlying object
   *
   */
  std::shared_ptr<nntrainer::Layer> &getObject();

  /**
   * @brief     Get underlying object
   *
   */
  const std::shared_ptr<nntrainer::Layer> &getObject() const;

  /**
   * @brief     Get index of the node
   *
   */
  size_t getIndex() const { return index; }

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool getTrainable() const noexcept;

  /**
   * Support interfaces for the properties intercepted from layer
   */

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  bool getFlatten() const { return flatten; }

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  bool getDistribute() const noexcept { return distribute; }

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  std::string getDistLayerType() const;

  /**
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActivationType getActivationType();

#ifdef PROFILE
  int event_key;
#endif

  /**
   * @brief   Overriding output stream for layers and it's derived class
   */
  friend std::ostream &operator<<(std::ostream &out, const LayerNode &l);

private:
  // TODO: make this unique_ptr once getObject API is removed
  std::shared_ptr<nntrainer::Layer>
    layer;      /**< The actual object in the graph node */
  size_t index; /**< index of each node */

  std::vector<std::string> input_layers;  /**< input layer names */
  std::vector<std::string> output_layers; /**< output layer names */
  bool flatten;    /**< flatten the output of this node */
  bool distribute; /**< to enable itearting along with time distribution */
  ActivationType
    activation_type; /**< activation applied to the output of this node */

  /**
   * These properties are set for the layer by the user but are intercepted
   * and used in the node which forms the basic element of the graph.
   */
  std::tuple<> props; /**< properties for the layer node */

  /**
   * @brief setProperty by PropertyType
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] type property type to be passed
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const nntrainer::Layer::PropertyType type,
                   const std::string &value = "");

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  const std::shared_ptr<nntrainer::Layer> &getLayer() const;

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  std::shared_ptr<nntrainer::Layer> &getLayer();
};

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] type Type of the layer to be constructed
 * @params[in] properties Properties of the layer
 */
std::unique_ptr<LayerNode>
createLayerNode(const std::string &type,
                const std::vector<std::string> &properties = {});

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] layer Already constructed layer
 * @params[in] properties Properties of the layer
 */
std::unique_ptr<LayerNode>
createLayerNode(std::shared_ptr<nntrainer::Layer> layer,
                const std::vector<std::string> &properties = {});

} // namespace nntrainer
#endif // __LAYER_NODE_H__
