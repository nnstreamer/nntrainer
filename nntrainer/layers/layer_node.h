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
 *
 * @details LayerNode provides a node wrapper around the Layer class to form a
 * GraphNode. Each layer is wrapped with LayerNode in order to add it to a
 * graph. Each LayerNode contains only 1 layer inside. LayerNode also intercepts
 * certain properties of the layer which are either related to graph related
 * connections (input_layers, output_layers, activation, flatten, distribute,
 * name) or essential for the description of the layer (trainable, input_dims)
 * iself. These properties, if needed by the layer object, are provided access
 * to via LayerContext.
 */

#ifndef __LAYER_NODE_H__
#define __LAYER_NODE_H__

#include <memory>
#include <tuple>
#include <vector>

#include <acti_func.h>
#include <graph_node.h>
#include <layer.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

class Layer;

class Exporter;
enum class ExportMethods;

namespace props {
class Name;
class Distribute;
class Flatten;
class ActivationType;
} // namespace props

/**
 * @class   LayerNode class
 * @brief   layer node class for the graph
 */
class LayerNode final : public ml::train::Layer, public GraphNode {
public:
  /**
   * @brief Default constructor
   */
  LayerNode() : LayerNode(nullptr, 0) {}

  /**
   * @brief Constructor of LayerNode class for v2
   * @param l layer to wrap with, the ownership is transferred to layer node
   *
   */
  LayerNode(std::unique_ptr<nntrainer::Layer> &&l, size_t idx = 0);

  /**
   * @brief     Destructor of LayerNode Class
   *
   */
  ~LayerNode();

public:
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
   * @param[in] properties values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name=property_val, ...}
   *
   *  @todo update to new signature: void setProperty(const
   * std::vector<std::string> &values)
   */
  int setProperty(std::vector<std::string> properties) override;

  /**
   * @brief     Get name of the layer
   *
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  const std::string getName() const noexcept override;

  /**
   * Support all the interface requirements by nntrainer::GraphNode
   */

  /**
   * @brief     Get index of the node
   *
   * @return    Index of the node
   */
  size_t getIndex() const { return index; }

  /**
   * @brief     Set the index for the node
   * @param     idx Index for the node
   */
  void setIndex(size_t idx) { index = idx; }

  /**
   * @brief     set name of layer
   *
   * @param[in] name Name of the layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   *
   * @todo update to new signature void setName(const std::string &name)
   */
  int setName(const std::string &name) { return setProperty({"name=" + name}); }

  /**
   * @brief     Get the input connections for this node
   *
   * @return list of name of the nodes which form input connections
   */
  const std::vector<std::string> &getInputConnections() const {
    return getInputLayers();
  }

  /**
   * Support all the interface requirements by nntrainer::Layer
   */

  /**
   * @brief     Finalize creating the layer node
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
  void finalize();

  /**
   * @brief     Forward Propagation of a layer
   * @param     training true if training, false if inference
   *
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void forwarding(bool training = true);

  /**
   * @brief     calc the derivative to be passed to the previous layer
   *
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void calcDerivative();

  /**
   * @brief     Calculate the derivative of a layer
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void calcGradient();

  /**
   * @brief this function helps exporting the layer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param     exporter exporter that conatins exporting logic
   * @param     method enum value to identify how it should be exported to
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const;

  /**
   * @brief Set the batch for the layer
   * @param     batch Batch value to be set
   * @details Update the run context based on the updated batch size if required
   */
  void setBatch(unsigned int batch);

  /**
   * @brief   If the current layer can support in-place
   * @return  true if inplace, else false
   */
  bool supportInPlace() const;

  /**
   * @brief  check if this layer requires label to be passed
   * @return true if requires a label when training, else false
   * @note   if requireLabel() == true means, for now, that it is endpoint of a
   * graph(numOutlayers == 0). label will be fed to the gradient of hidden if
   * requireLabel is true
   */
  bool requireLabel() const;

  /**
   * Add rest of the helper interfaces required by other internal classes
   */

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool supportBackwarding() const noexcept {
    return getLayer()->supportBackwarding();
  }

  /**
   * Support interfaces for the properties intercepted from layer
   */

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool getTrainable() const noexcept;

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  bool getFlatten() const noexcept;

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  bool getDistribute() const noexcept;

  /**
   * @brief     get activation for this layer
   * @retval dist to enable/disable distribute
   */
  ActivationType getActivationToBeRealized() const noexcept;

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  std::string getDistLayerType() const;

  /**
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActivationType getActivationType() const;

  /**
   * @brief     Get number of input connections
   * @retval    number of inputs
   */
  unsigned int getNumInputConnections() const { return input_layers.size(); }

  /**
   * @brief     Get number of output connections
   * @retval    number of outputs
   */
  unsigned int getNumOutputConnections() const { return output_layers.size(); }

  /**
   * @brief     Get number of inputs
   * @retval    number of inputs
   */
  unsigned int getNumInputs() const { return init_context.getNumInputs(); }

  /**
   * @brief     Get number of outputs
   * @retval    number of outputs
   */
  unsigned int getNumOutputs() const { return init_context.getNumOutputs(); }

  /**
   * @brief Get the number of weights
   *
   * @return unsigned int number of weights
   */
  unsigned int getNumWeights() const { return init_context.getNumWeights(); }

  /**
   * @brief     Get the Input Layers object
   *
   * @return const std::vector<std::string>&
   */
  const std::vector<std::string> &getInputLayers() const {
    return input_layers;
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
    resizeInputDimensions(input_layers.size());
  }

  /**
   * @brief Add name to the output layers
   *
   * @param out_layer Name to be added
   */
  void addOutputLayers(const std::string &out_layer) {
    output_layers.push_back(out_layer);
    init_context =
      InitLayerContext(init_context.getInputDimensions(), output_layers.size());
  }

  /**
   * @brief Set the Input Layers object
   *
   * @param layers Name of the layers
   */
  void setInputLayers(const std::vector<std::string> &layers) {
    input_layers = layers;
    resizeInputDimensions(input_layers.size());
  }

  /**
   * @brief Set the Output Layers object
   *
   * @param layers Name of the layers
   */
  void setOutputLayers(const std::vector<std::string> &layers) {
    output_layers = layers;
    init_context =
      InitLayerContext(init_context.getInputDimensions(),
                       std::max((unsigned int)output_layers.size(), 1u));
  }

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  const std::vector<TensorDim> getInputDimensions() const {
    return init_context.getInputDimensions();
  }

  /**
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  const std::vector<TensorDim> getOutputDimensions() const {
    return init_context.getOutputDimensions();
  }

  /**
   * @brief Get the Weight object
   *
   * @param idx Identifier of the weight
   * @return Weight& Reference to the weight
   */
  Weight getWeightWrapper(unsigned int idx) {
    if (run_context.weightHasGradient(idx)) {
      return Weight(run_context.getWeight(idx), run_context.getWeightGrad(idx),
                    run_context.getWeightName(idx));
    } else {
      return Weight(run_context.getWeight(idx), Tensor(),
                    run_context.getWeightName(idx));
    }
  }

  /**
   * @brief Get the Weight object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Weight &getWeightObject(unsigned int idx) {
    return run_context.getWeightObject(idx);
  }

  /**
   * @brief Get the Weight tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Tensor &getWeight(unsigned int idx) { return run_context.getWeight(idx); }

  /**
   * @brief Get the Weight Gradient tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightGrad(unsigned int idx) {
    return run_context.getWeightGrad(idx);
  }

  /**
   * @brief Get the Weight object name
   *
   * @param idx Identifier of the weight
   * @return const std::string &Name of the weight
   */
  const std::string &getWeightName(unsigned int idx) {
    return run_context.getWeightName(idx);
  }

  /**
   * @brief Get the Input tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInput(unsigned int idx) { return run_context.getInput(idx); }

  /**
   * @brief Get the Input Grad tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInputGrad(unsigned int idx) {
    return run_context.getInputGrad(idx);
  }

  /**
   * @brief Get the Output tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output tensor
   */
  Tensor &getOutput(unsigned int idx) { return run_context.getOutput(idx); }

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  Tensor &getOutputGrad(unsigned int idx) {
    return run_context.getOutputGrad(idx);
  }

  /**
   * @brief Get the Output Grad unsafe
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  Tensor &getOutputGradUnsafe(unsigned int idx) {
    return run_context.getOutputGradUnsafe(idx);
  }

  /**
   * @brief     read layer Weight & Bias data from file
   * @param file input file stream
   */
  void read(std::ifstream &file);

  /**
   * @brief     save layer Weight & Bias data from file
   * @param file output file stream
   */
  void save(std::ofstream &file) const;

  /**
   * @brief     get loss for the layer
   * @return    loss of the layer
   *
   * @todo      Update this for loss layer
   */
  float getLoss() const {
    float loss = run_context.getLoss();
    for (unsigned int idx = 0; idx < run_context.getNumWeights(); idx++) {
      loss += run_context.getWeightRegularizationLoss(idx);
    }

    return loss;
  }

#ifdef PROFILE
  int event_key;
#endif

  /**
   * @brief   Overriding output stream for layers and it's derived class
   */
  friend std::ostream &operator<<(std::ostream &out, const LayerNode &l);

  /**
   * @brief   Get init layer context
   *
   * @retval  init layer context
   */
  const InitLayerContext &getInitContext() const { return init_context; }

  /**
   * @brief   Get run layer context
   *
   * @retval  run layer context
   */
  const RunLayerContext &getRunContext() const { return run_context; }

  /**
   * @brief   Set run layer context
   *
   * @param  context Updated run layer context
   */
  void updateRunContext(RunLayerContext &&context) {
    // TODO: ensure props/trainable must match
    run_context = std::move(context);
  }

  /**
   * @brief Set input dimension for the layer
   *
   * @param dim Input tensor dim
   * @param idx Index of the dim
   */
  void setInputDimension(const TensorDim &dim, unsigned int idx) {
    if (idx >= getNumInputs())
      throw std::out_of_range("Setting dimensions out of bounds");

    std::vector<TensorDim> input_dim = init_context.getInputDimensions();
    if (input_dim[idx] != dim) {
      input_dim[idx] = dim;
      init_context = InitLayerContext(input_dim, init_context.getNumOutputs());
    }
  }

private:
  std::unique_ptr<nntrainer::Layer>
    layer; /**< The actual object in the graph node */

  // TODO: possibly remove, two identifiers for the same node (name and
  // index) can lead to issues later
  size_t index;   /**< index of each node */
  bool finalized; /**< if the layer node has been finalized */

  std::vector<std::string> input_layers;  /**< input layer names */
  std::vector<std::string> output_layers; /**< output layer names */
  ActivationType
    activation_type; /**< activation applied to the output of this node */

  InitLayerContext init_context; /**< context to be built for/while
                                    initialization of the layer. This will also
                                    contain the properties of the layer. */

  RunLayerContext run_context; /**< context required for running/execution of
                    the layer. This will also contain the properties of the
                    layer. The properties will be copied upon final creation.
                    Editing properties of the layer after init will not the
                    properties in the context/graph unless intended. */

  using PropsType = std::tuple<props::Name, props::Flatten, props::Distribute,
                               props::Trainable>;
  /**
   * These properties are set for the layer by the user but are intercepted
   * and used in the node which forms the basic element of the graph.
   */
  std::unique_ptr<PropsType> layer_node_props; /**< properties for the node */

  /**
   * @brief setProperty by PropertyType
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] key property type to be passed
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @return true if the property can be captured, else false
   * @exception std::invalid_argument invalid argument
   */
  bool setProperty(const std::string &key, const std::string &value);

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  const nntrainer::Layer *getLayer() const;

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  nntrainer::Layer *getLayer();

  /**
   * @brief     Activation Setter
   * @param[in] activation activation type
   * @throw std::invalid_argument when ActivationType is unknown
   */
  void setActivation(ActivationType activation);

  /**
   * @brief     Resize the input dimensions
   *
   * @param size Number of input dimensions
   */
  void resizeInputDimensions(unsigned int size) {
    auto cur_input_dim = init_context.getInputDimensions();
    if (cur_input_dim.size() != size) {
      cur_input_dim.resize(size);
      init_context =
        InitLayerContext(cur_input_dim, init_context.getNumOutputs());
    }
  }
};

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] type Type of the layer to be constructed
 * @params[in] properties Properties of the layer
 */
std::unique_ptr<LayerNode>
createLayerNode(const ml::train::LayerType &type,
                const std::vector<std::string> &properties = {});

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
createLayerNode(std::unique_ptr<nntrainer::Layer> &&layer,
                const std::vector<std::string> &properties);

} // namespace nntrainer
#endif // __LAYER_NODE_H__
