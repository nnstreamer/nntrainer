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

#define LAYER_V2 false

#ifndef __LAYER_NODE_H__
#define __LAYER_NODE_H__

#include <graph_node.h>
#include <layer.h>
#include <layer_context.h>
#include <layer_internal.h>
#include <node_exporter.h>

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
  LayerNode(std::shared_ptr<nntrainer::LayerV1> l, size_t idx = 0);

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
  const std::string getName() const noexcept override {
    return std::get<props::Name>(props).get();
  }

  /**
   * Support all the interface requirements by GraphNode<nntrainer::Layer>
   */

  /**
   * @brief     Get underlying object
   *
   */
  std::shared_ptr<nntrainer::LayerV1> &getObject();

  /**
   * @brief     Get underlying object
   *
   */
  const std::shared_ptr<nntrainer::LayerV1> &getObject() const;

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

  /**
   * @brief     Get number of inputs
   * @retval    number of inputs
   */
  unsigned int getNumInputs() const { return input_layers.size(); }

  /**
   * @brief     Get number of outputs
   * @retval    number of outputs
   */
  unsigned int getNumOutputs() const { return output_layers.size(); }

  /**
   * @brief     Get the Input Layers object
   *
   * @return const std::vector<std::string>&
   */
  const std::vector<std::string> &getInputLayers() const {
    return input_layers;
  }

  /**
   * @brief     Get the input connections for this node
   *
   * @return list of name of the nodes which form input connections
   */
  const std::vector<std::string> &getInputConnections() const {
    return getInputLayers();
  }

  /**
   * @brief Get the Output Layers object
   *
   * @return const std::vector<std::string>&
   */
  const std::vector<std::string> &getOutputLayers() const {
    return output_layers;
  }

  /**
   * @brief Update input layers entry name
   *
   * @param from The name to be updated
   * @param to The name to be updated to
   */
  void updateInputLayers(const std::string &from, const std::string &to);

  /**
   * @brief Update the input layers name at the given idx
   *
   * @param idx The index at which layer name must be updated
   * @param to The name to be updated to
   */
  void updateInputLayers(const unsigned int idx, const std::string &to);

  /**
   * @brief Add name to the input layers
   *
   * @param in_layer Name to be added
   */
  void addInputLayers(const std::string &in_layer) {
    input_layers.push_back(in_layer);
    layer->setNumInputs(input_layers.size());
  }

  /**
   * @brief Add name to the output layers
   *
   * @param out_layer Name to be added
   */
  void addOutputLayers(const std::string &out_layer) {
    output_layers.push_back(out_layer);
    layer->setNumOutputs(output_layers.size());
  }

  /**
   * @brief Set the Input Layers object
   *
   * @param layers Name of the layers
   */
  void setInputLayers(const std::vector<std::string> &layers) {
    input_layers = layers;
    layer->setNumInputs(layers.size());
  }

  /**
   * @brief Set the Output Layers object
   *
   * @param layers Name of the layers
   */
  void setOutputLayers(const std::vector<std::string> &layers) {
    output_layers = layers;
    layer->setNumOutputs(layers.size());
  }

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  const std::vector<TensorDim> getInputDimensions() const {
    if (LAYER_V2) {
      return init_context.getInputDimensions();
    } else {
      return getLayer()->getInputDimension();
    }
  }

  /**
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  const std::vector<TensorDim> getOutputDimensions() const {
    if (LAYER_V2) {
      return init_context.getOutputDimensions();
    } else {
      return getLayer()->getOutputDimension();
    }
  }

  /**
   * @brief this function helps exporting the layer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param exporter exporter that conatins exporting logic
   * @param method enum value to identify how it should be exported to
   */
  virtual void
  export_to(Exporter &exporter,
            ExportMethods method = ExportMethods::METHOD_STRINGVECTOR) const {
    exporter.saveResult(props, method, this);
    layer->export_to(exporter, method);
  }

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file);

  /**
   * @brief     save layer Weight & Bias data from file
   * @param[in] file output file stream
   */
  void save(std::ofstream &file) const;

#ifdef PROFILE
  int event_key;
#endif

  /**
   * @brief   Overriding output stream for layers and it's derived class
   */
  friend std::ostream &operator<<(std::ostream &out, const LayerNode &l);

private:
  // TODO: make this unique_ptr once getObject API is removed
  std::shared_ptr<nntrainer::LayerV1>
    layer; /**< The actual object in the graph node */
  // TODO: possibly remove, two identifiers for the same  node (name and index)
  // can lead to issues later
  size_t index; /**< index of each node */

  /** TODO : move management of num_inputs to layer_node */
  std::vector<std::string> input_layers;  /**< input layer names */
  std::vector<std::string> output_layers; /**< output layer names */
  bool flatten;    /**< flatten the output of this node */
  bool distribute; /**< to enable itearting along with time distribution */
  ActivationType
    activation_type; /**< activation applied to the output of this node */

  RunLayerContext
    run_context; /**< context required for running/execution of the layer. This
                    will also contain the properties of the layer. The
                    properties will be copied upon final creation. Editing
                    properties of the layer after init will not the properties
                    in the context/graph unless intended. */
  InitLayerContext init_context; /**< context to be built for/while
                                    initialization of the layer. This will also
                                    contain the properties of the layer. */
  /**
   * These properties are set for the layer by the user but are intercepted
   * and used in the node which forms the basic element of the graph.
   */
  std::tuple<props::Name> props; /**< properties for the layer node */

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
  void setProperty(const nntrainer::LayerV1::PropertyType type,
                   const std::string &value = "");

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  const std::shared_ptr<nntrainer::LayerV1> &getLayer() const;

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  std::shared_ptr<nntrainer::LayerV1> &getLayer();
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
createLayerNode(std::shared_ptr<nntrainer::LayerV1> layer,
                const std::vector<std::string> &properties = {});

} // namespace nntrainer
#endif // __LAYER_NODE_H__
