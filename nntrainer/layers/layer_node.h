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
 * connections (input_connections, output_connections, activation, flatten,
 * distribute, name) or essential for the description of the layer (trainable,
 * input_dims) iself. These properties, if needed by the layer object, are
 * provided access to via LayerContext.
 */

#ifndef __LAYER_NODE_H__
#define __LAYER_NODE_H__

#include <memory>
#include <tuple>
#include <vector>

#include <graph_node.h>
#include <layer.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <weight.h>

namespace nntrainer {

class Layer;
class Connection;
class Exporter;
enum class ExportMethods;

namespace props {
class Name;
class Distribute;
class Flatten;
class Loss;
class InputShape;
class Activation;
class SharedFrom;
class InputConnection;
class ClipGradByGlobalNorm;
} // namespace props

/**
 * @brief Enum class for the various types of inplace modes supported by layer
 *
 */
enum class InPlace {
  NONE,           /**< layer is not inplace */
  RESTRICTING,    /**< layer is in-place and does place restriction on layers
                    ahead of it to be in-place */
  NON_RESTRICTING /**< layer is in-place and does NOT place restriction on the
                    layers ahead of it to be in-place */
};

/**
 * @class   LayerNode class
 * @brief   layer node class for the graph
 */
class LayerNode final : public ml::train::Layer, public GraphNode {
public:
  /**
   * @brief Constructor of LayerNode class for v2
   * @param l layer to wrap with, the ownership is transferred to layer node
   *
   */
  LayerNode(std::unique_ptr<nntrainer::Layer> &&l);

  /**
   * @brief     Destructor of LayerNode Class
   *
   */
  ~LayerNode();

  /**
   * Support all the interface requirements by ml::train::Layer
   */

  /**
   * @brief Get the layer type
   *
   * @return const std::string type representation
   */
  const std::string getType() const override;

  /**
   * @brief     set Property of layer
   * @param[in] properties values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name=property_val, ...}
   *
   */
  void setProperty(const std::vector<std::string> &properties) override;

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
   * @brief     set name of layer
   *
   * @param[in] name Name of the layer
   */
  void setName(const std::string &name) override {
    setProperty({"name=" + name});
  }

  /**
   * @brief Get the Input Connection Index object
   *
   * @param nth nth input
   * @throws if nth is out of range of getNumInputConnection()
   * @return const unsigned index
   */
  const unsigned getInputConnectionIndex(unsigned nth) const;

  /**
   * @brief Get the Input Connection Name object
   *
   * @param nth nth input
   * @throws if nth is out of range of getNumInputConnection()
   * @return const std::string& name
   */
  const std::string &getInputConnectionName(unsigned nth) const;

  /**
   * @brief Set the Input Connection Index object
   *
   * @param nth nth input
   * @param index index to set
   * @throws if nth is out of range of getNumInputConnection()
   */
  void setInputConnectionIndex(unsigned nth, unsigned index);

  /**
   * @brief Get the Input Connection Name object
   *
   * @param nth input
   * @param index index to set
   * @throws if nth is out of range of getNumInputConnection()
   * @throws if new identifier is invalid
   */
  void setInputConnectionName(unsigned nth, const std::string &name);

  /**
   * @brief Get the output connection object
   *
   * @param nth nth input
   * @throws if nth is out of range of getNumOutputConnection()
   * @return Connection * view of a connection, null means this does not exist
   */
  const Connection *getOutputConnection(unsigned nth) const;

  /**
   * @brief Set the Output Connection
   * @note Each output must be identified only ONCE.
   * @note when nth comes, getNumOutput() expends to nth + 1 as resize occurs.
   * Please also notice none identified intermediate output (or mismatch between
   * actual number of out tensors and output) is allowed but will produce
   * warning, this implies that the output is not used else where.
   * @throw std::invalid_argument when trying to identify output
   * more then once
   *
   * @param nth nth output
   * @param name name of the output bound connection
   * @param index index of the output bound connection
   */
  void setOutputConnection(unsigned nth, const std::string &name,
                           unsigned index);

  /**
   * @brief     Get the input connections for this node
   *
   * @return list of name of the nodes which form input connections
   */
  const std::vector<std::string> getInputConnections() const override {
    return getInputLayers();
  }

  /**
   * @brief     Get the output connections for this node
   *
   * @return list of name of the nodes which form output connections
   */
  const std::vector<std::string> getOutputConnections() const override {
    return getOutputLayers();
  }

  /**
   * @brief     get the execution order/location of this node
   *
   * @retval    the execution order/location of this node
   */
  ExecutionOrder getExecutionOrder() const override { return exec_order; }

  /**
   * @brief     set the execution order/location of this node
   *
   * @param     exec_order the execution order/location of this node
   */
  void setExecutionOrder(ExecutionOrder exec_order_) override {
    exec_order = exec_order_;
  }

  /**
   * Support all the interface requirements by nntrainer::Layer
   */

  /**
   * @brief     Finalize creating the layer node
   *
   * @param   input_dims input dimension provided to be used to set output
   * dimensions. if empty function This function must set output dimensions in
   * the given context. Further, context can be used to request weights for the
   * layer, and any extra tensor required for the operation of the layer.
   * @note      After calling this it is not allowed to
   * change properties.
   * @note      No memory allocation must be performed in the initialization
   * step. Any tensor memory required must be requested to the context which
   * will be made available during execution of the layer with the context.
   * @note configureRunContext() is expected to called right after this.
   */
  InitLayerContext finalize(const std::vector<TensorDim> &input_dims = {});

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
   * @brief   Notify that this layer will execute in-place
   *
   * @param val in place state for the layer
   */
  void executeInPlace(InPlace val) {
    if (val != InPlace::NONE && !supportInPlace())
      throw std::runtime_error("Error setting layer to work in-place");

    inplace = val;
  }

  /**
   * @brief   Get if the layer is going to execute in-place
   *
   * @return InPlace type for the layer
   */
  InPlace executeInPlace() const { return inplace; }

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
  bool supportBackwarding() const { return getLayer()->supportBackwarding(); }

  /**
   * Support interfaces for the properties intercepted from layer
   */

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool getTrainable() const;

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  bool getFlatten() const;

  /**
   * @brief Get the Shared From property of the layer node
   *
   * @return std::string node name where the weights are borrowed
   */
  std::string getSharedFrom() const;

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  bool getDistribute() const;

  /**
   * @brief     get activation for this layer
   * @retval dist to enable/disable distribute
   */
  ActivationType getActivationToBeRealized() const;

  /**
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActivationType getActivationType() const;

  /**
   * @brief     Get number of input connections
   * @retval    number of inputs
   */
  unsigned int getNumInputConnections() const;

  /**
   * @brief     Get number of output connections
   * @retval    number of outputs
   */
  unsigned int getNumOutputConnections() const;

  /**
   * @brief     Get number of inputs
   * @retval    number of inputs
   */
  unsigned int getNumInputs() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getNumInputs();
  }

  /**
   * @brief     Get number of outputs
   * @retval    number of outputs
   */
  unsigned int getNumOutputs() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getNumOutputs();
  }

  /**
   * @brief Get the number of weights
   *
   * @return unsigned int number of weights
   */
  unsigned int getNumWeights() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getNumWeights();
  }

  /**
   * @brief Set the Output Layers object
   *
   * @param layers Name of the layers
   */
  void setOutputLayers(const std::vector<std::string> &layers);

  /**
   * @brief check if input shape property is set
   *
   * @return bool true if input shape property has set
   */
  bool hasInputShapeProperty() const;

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  const std::vector<TensorDim> getInputDimensions() const;

  /**
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  const std::vector<TensorDim> getOutputDimensions() const;
  /**
   * @brief Get the Weight object
   *
   * @param idx Identifier of the weight
   * @return Weight& Reference to the weight
   */
  Weight getWeightWrapper(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    if (run_context->weightHasGradient(idx)) {
      return Weight(run_context->getWeight(idx),
                    run_context->getWeightGrad(idx),
                    run_context->getWeightName(idx));
    } else {
      return Weight(run_context->getWeight(idx), Tensor(),
                    run_context->getWeightName(idx));
    }
  }

  /**
   * @brief Get the Weight object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Weight &getWeightObject(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeightObject(idx);
  }

  /**
   * @brief Get the Weight tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Tensor &getWeight(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeight(idx);
  }

  /**
   * @brief Get the Weight Gradient tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightGrad(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeightGrad(idx);
  }

  /**
   * @brief Get the Weight object name
   *
   * @param idx Identifier of the weight
   * @return const std::string &Name of the weight
   */
  const std::string &getWeightName(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeightName(idx);
  }

  /**
   * @brief Get the Input tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInput(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getInput(idx);
  }

  /**
   * @brief Get the Input Grad tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInputGrad(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getInputGrad(idx);
  }

  /**
   * @brief Get the Output tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output tensor
   */
  Tensor &getOutput(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getOutput(idx);
  }

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  Tensor &getOutputGrad(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getOutputGrad(idx);
  }

  /**
   * @brief Get the Output Grad unsafe
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  Tensor &getOutputGradUnsafe(unsigned int idx) {
    return run_context->getOutputGradUnsafe(idx);
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
   */
  float getLoss() const;

#ifdef PROFILE
  int forward_event_key;
  int calc_deriv_event_key;
  int calc_grad_event_key;
#endif

  /**
   * @brief   Overriding output stream for layers and it's derived class
   */
  friend std::ostream &operator<<(std::ostream &out, const LayerNode &l);

  /**
   * @brief   Get run layer context
   *
   * @retval  run layer context
   */
  RunLayerContext &getRunContext() {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return *run_context;
  }

  /**
   * @brief   Get run layer context
   *
   * @retval  run layer context
   */
  const RunLayerContext &getRunContext() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return *run_context;
  }

  /**
   * @brief   check if layer is finalized
   *
   * @retval  bool true if the layer is finalized else false
   */
  bool isFinalized() const {
    if (!run_context)
      return false;

    return true;
  }

  /**
   * @brief Set the Run Context object with given tensor packs
   *
   * @param weights weights
   * @param inputs inputs
   * @param outputs outputs
   * @param tensors tensors
   */
  void configureRunContext(const std::vector<Weight *> &weights,
                           const std::vector<Var_Grad *> &inputs,
                           const std::vector<Var_Grad *> &outputs,
                           const std::vector<Var_Grad *> &tensors);

  /**
   * @brief Preset modes for printing summary for the layer
   */
  enum class PrintPreset {
    PRINT_NONE = 0,     /**< Print nothing */
    PRINT_SUMMARY,      /**< Print preset including summary information */
    PRINT_SUMMARY_META, /**< Print summary preset that includes meta information
                         */
    PRINT_ALL           /**< Print everything possible */
  };

  /**
   * @brief print using PrintPreset
   *
   * @param out oustream
   * @param preset preset to be used
   */
  void printPreset(std::ostream &out,
                   PrintPreset preset = PrintPreset::PRINT_SUMMARY);

  /**
   * @brief remap identifier inside layer node
   *
   * @param remap_fn function to remap
   */
  void remapIdentifiers(std::function<void(std::string &)> remap_fn);

  /**
   * @brief remap connections(input, output layers ) inside layer node
   *
   * @param remap_fn function to remap
   */
  void
  remapConnections(std::function<void(std::string &, unsigned &)> remap_fn);

  /**
   * @brief create the same node with same properties and types
   *
   * @note this must be done before finalize() as finalize has some potential to
   * change some properties
   * @return LayerNode newly created node
   */
  std::unique_ptr<LayerNode> cloneConfiguration();

  /**
   * @brief Set if the layer needs to do derivative calculation
   *
   * @param nb true if the layer needs to do backwarding, else false
   */
  void needsCalcDerivative(bool nb) {
    NNTR_THROW_IF(nb && !supportBackwarding(), std::invalid_argument)
      << "[Layer] " << getName()
      << " does not support backwarding but is needed";
    needs_calc_derivative = nb;
  }

  /**
   * @brief Set if the layer needs to do calculation of gradients
   *
   * @param nb true if the layer needs to do backwarding, else false
   */
  void needsCalcGradient(bool nb) { needs_calc_gradient = nb; }

  /**
   * @brief Get the layer needs to do calculation of derivatives
   *
   * @return true if the layer needs to do backwarding, else false
   */
  bool needsCalcDerivative() { return needs_calc_derivative; }

  /**
   * @brief Set if the layer needs to do calculation of gradient
   *
   * @param nb true if the layer needs to do backwarding, else false
   */
  bool needsCalcGradient() { return needs_calc_gradient; }

private:
  /**
   * @brief     Get the Input Layers object
   *
   * @return const std::vector<std::string>
   */
  const std::vector<std::string> getInputLayers() const;

public:
  /**
   * @brief Get the Output Layers object
   *
   * @return const std::vector<std::string>
   */
  const std::vector<std::string> getOutputLayers() const;

  std::unique_ptr<nntrainer::Layer>
    layer; /**< The actual object in the graph node */

  InPlace
    inplace; /**< store if the current layer is going to operate in-place */
  bool needs_calc_derivative; /**< cache if this layer needs to do
                                 calcDerivative */
  bool needs_calc_gradient; /**< cache if this layer needs to do calcGradient */

  std::vector<std::unique_ptr<Connection>>
    output_connections;                      /**< output layer names */
  unsigned effective_output_connection_size; /**< effective output connection
                                                size, this skips not connected
                                                slot, so this number can be
                                                diffrent from num_outputs() */

  std::unique_ptr<RunLayerContext>
    run_context; /**< context required for running/execution of the layer. This
will also contain the properties of the layer. The properties will be copied
upon final creation. Editing properties of the layer after init will not the
properties in the context/graph unless intended. */

  using PropsType = std::tuple<props::Name, props::Distribute, props::Trainable,
                               std::vector<props::InputConnection>,
                               std::vector<props::InputShape>,
                               props::SharedFrom, props::ClipGradByGlobalNorm>;

  using RealizationPropsType = std::tuple<props::Flatten, props::Activation>;
  /** these realization properties results in addition of new layers, hence
   * skipped in generation of model architecture as the correspondingly layer
   * itself is added. Distribute is also a property which is realized, but as it
   * doesn't add new layer, it is saved. */

  /**
   * These properties are set for the layer by the user but are intercepted
   * and used in the node which forms the basic element of the graph.
   */
  std::unique_ptr<PropsType> layer_node_props; /**< properties for the node */
  std::unique_ptr<RealizationPropsType>
    layer_node_props_realization;    /**< properties for the node */
  std::unique_ptr<props::Loss> loss; /**< loss */
  float regularization_loss;
  ExecutionOrder exec_order; /**< order/location of execution for this node
                                   in forward and backwarding operations */

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
   * @brief anchor point to override if PRINT_SHAPE_INFO is enabled for
   * Layer::print()
   */
  void printShapeInfo(std::ostream &out);

  /**
   * @brief anchor point to override if PRINT_METRIC is enabled for
   * Layer::print()
   */
  void printMetric(std::ostream &out);

  /**
   * @brief     Print layer related information. Do not override without clear
   * reason. It is recommended to override printShapeInfo, printPropertiesMeta,
   * printProperties, printMetric instead
   * @param[in] out outstream
   * @param[in] flags combination of LayerPrintOption
   */
  void print(std::ostream &out, unsigned int flags = 0);
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
