// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network Graph Class for Neural Network
 *
 */

#ifndef __NETWORK_GRAPH_H__
#define __NETWORK_GRAPH_H__
#ifdef __cplusplus

#include <list>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include <common_properties.h>
#include <compiler_fwd.h>
#include <graph_core.h>
#include <layer_node.h>
#include <manager.h>
#include <model_common_properties.h>
#include <optimizer_wrapped.h>
#include <subgraph.h>

namespace nntrainer {

using ExecutionMode = ml::train::ExecutionMode;

class Connection;
/**
 * @class   NeuralNetwork Graph Class
 * @brief   NeuralNetwork Graph Class which manage layers
 */
class NetworkGraph {

public:
  /**
   * @brief     Constructor of NeuralNetwork Graph Class
   */
  NetworkGraph() :
    tensor_manager(std::make_shared<Manager>()),
    graph(),
    compiled(false),
    batch_size(0),
    graph_exec_end(0),
    backward_iter_end(nullptr),
    forward_iter_end(nullptr),
    optimize_memory(true),
    exec_mode(ExecutionMode::TRAIN),
    tensor_format("NCHW"),
    tensor_dtype_str("FP32-FP32"),
    tensor_dtype(split("FP32-FP32", getRegex("\\-"))),
    is_clip_grad(false),
    loss_scale(1.0f),
    lookahead(0) {
    nan_count = 0;

    /**
     * @note NetworkGraph constructor without any parameters.
     * This constructor creates a `default_graph` to handle general scenarios.
     * If the default subgraph is not utilized, it will be discarded at the
     * compilation time
     */
    auto sg = std::make_shared<SubGraphCpu>(tensor_manager);
    sg->setName("default");
    graph.addNode(SGNODE(sg));
  }

  /**
   * @brief     Constructor of NeuralNetwork Graph Class, which is invoked at
   * compile time.
   * The `NetworkGraph` class constructor initializes the object using the
   * provided graph representation. It integrates both inter-subgraph and
   * intra-subgraph representation. Finally, it returns the constructed
   * layer-node level representation to `graph_ln_representation`.
   * @param[in] enable_swap enable memory swap for tensor
   * @param[in] model_props model property fixed at the compile time
   * @param[in] graph_representation graph representation to initialize
   * NetworkGraph
   * @param[in] graph_ln_representation graph layer node representation to be
   * updated by this constructor.
   * @param[in] mode execution mode (default ExecutionMode::TRAIN)
   * @param[in] swap_path memory swap file path when the swap is enabled
   * @param[in] tensor_format define tensor format. One of NCHW and NHWC
   * (default NCHW)
   * @param[in] tensor_dtype_ It says weight type and activation type (default
   * FP32-FP32)
   */
  NetworkGraph(bool enable_swap, const ModelPropsType &model_props,
               GraphRepresentation &graph_representation,
               GraphLayerNodeRepresentation &graph_ln_representation,
               ExecutionMode mode = ExecutionMode::TRAIN,
               const std::string &swap_path = "", unsigned int lookahead = 0,
               const std::string &tensor_format_ = "NCHW",
               const std::string &tensor_dtype_ = "FP32-FP32") :
    tensor_manager(std::make_shared<Manager>(
      enable_swap, swap_path, lookahead, tensor_format_, tensor_dtype_, mode)),
    graph(),
    compiled(false),
    batch_size(0),
    graph_exec_end(0),
    backward_iter_end(nullptr),
    forward_iter_end(nullptr),
    optimize_memory(true),
    exec_mode(mode),
    tensor_format(tensor_format_),
    tensor_dtype_str(tensor_dtype_),
    tensor_dtype(split(tensor_dtype_, getRegex("\\-"))),
    is_clip_grad(false),
    loss_scale(1.0f),
    lookahead(lookahead) {
    nan_count = 0;
    graph_ln_representation.clear();

    /**
     * @note If no layers are added. Create a default graph for dummy
     * otherwise, the subgraphs are created based on the graph_representaiton
     * info.
     */
    if (!graph_representation.size()) {
      auto sg = std::make_shared<SubGraphCpu>(tensor_manager, mode, lookahead,
                                              tensor_format_, tensor_dtype_);
      sg->setName("default");
      graph.addNode(SGNODE(sg));
      graph_representation.push_back(sg);
      return;
    }

    realize(model_props, graph_representation, graph_ln_representation);
  }

  /**
   * @brief   Destructor of the NeuralNetwork Graph class
   *
   */
  ~NetworkGraph() = default;

  /**
   * @brief     Realize the network's internal layers
   */
  void realize(const ModelPropsType &model_props,
               GraphRepresentation &graph_representation,
               GraphLayerNodeRepresentation &graph_ln_representation);

  /**
   * @brief     Compile the graph
   * @param[in] loss_type loss for the graph
   * returns ML_ERROR_NONE on success, error on failure
   */
  int compile(const std::string &loss_type);

  /**
   * @brief Add a subgraph to `NetworkGraph`, setting the network graph
   * information within the subgraph.
   * @param[in] subgraph shared_ptr of SubGraph
   */
  void addSubGraph(const SubGraphNode subgraph);

  /**
   * @brief Create new LayerNode and add into Graph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayer(std::shared_ptr<LayerNode> layer);

  /**
   * @brief get current flat graph from the model before sorting
   * @note graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current flat graph
   *
   * @todo remove getting unsorted layers from model loader, compile model
   * loader
   */
  std::vector<std::shared_ptr<LayerNode>>
  getUnsortedLayers(const std::string &input_layer,
                    const std::string &output_layer) const;

  /**
   * @brief getter of number of all layer nodes
   * @param[out] number of layer nodes
   */
  unsigned int size() const {
    unsigned int size = 0;
    for (auto it = cbegin(); it != cend(); ++it)
      size += (*it)->size();
    return size;
  }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const {
    bool is_empty = true;
    for (auto it = cbegin(); it != cend(); ++it)
      is_empty = (is_empty && (*it)->empty());
    return is_empty;
  }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(NetworkGraph &lhs, NetworkGraph &rhs) {
    /// @fixme this swap function need maintenance
    using std::swap;

    swap(lhs.graph, rhs.graph);
  }

  /**
   * @brief getter of Sorted LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  std::shared_ptr<LayerNode> getSortedLayerNode(unsigned int ith) const {
    /**
     * @note This code written based on the assumption that he graph consists
     * with only one default subgraph node. It needs to be updated.
     * @todo update the code to consider `ith` as a global layer node index.
     */
    return (*cbegin())->getSortedLayerNode(ith);
  }

  /**
   * @brief getter of LayerNode with layer name
   * @param[in] layer name
   * @retval LayerNode
   */
  std::shared_ptr<LayerNode> getLayerNode(const std::string &layer_name) const {
    std::shared_ptr<LayerNode> ln;
    for (auto it = cbegin(); it != cend(); ++it) {
      ln = it->getLayerNode(layer_name);
      if (ln)
        return ln;
    }
    return ln;
  }

  /**
   * @brief getter all the layer nodes in the model
   * @retval Layer nodes
   * @note these layer nodes will be in sorted order if the model is compiled,
   * otherwise the order is the order of addition of layer nodes in the model.
   */
  std::vector<std::shared_ptr<LayerNode>> getLayerNodes() const;

  /**
   * @brief     set batch size
   * @param[in] batch size
   */
  void setBatchSize(unsigned int batch_size);

  /**
   * @brief try apply gradient if possible
   * @note if it is not the last of the gradient access, this is noop
   * @note if the gradient is to be clipped by norm, this is noop
   *
   * @param node node to try apply gradient
   * @param iteration iteration where the applyGradients is called
   * @param opt shared ptr of the optimizer used for applyGradients (opt is
   * passed from neuralnetwork)
   */
  void applyGradients(LayerNode *node, int iteration,
                      std::shared_ptr<OptimizerWrapped> opt);

  /**
   * @brief     forwarding network graph
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors forwarding(
    bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool swap_mode = false);

  /**
   * @brief     forwarding network graph
   * @param[in] from start step
   * @param[in] to end step
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors incremental_forwarding(
    unsigned int from, unsigned int to, bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr);

  /**
   * @brief     backwarding the network graph
   * @param[in] iteration current iteration number
   * @param[in] stop_cb callback function which return stop condition
   * @param[in] user_data user data used for backwarding
   * @param[in] is_grad_opt_mode flag to designate grad_opt_mode (passed from
   *            neuralnet)
   * @param[in] opt shared ptr of the optimizer used for applyGradients (opt is
   *            passed from neuralnetwork)
   * @retval ret it is false then the gradient has NaN valude in mixed precision
   *         training. If it is, then we need to control the loss scale factor
   *         and compute again the derivatives.
   */
  bool backwarding(
    int iteration,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool is_grad_opt_mode = false,
    std::shared_ptr<OptimizerWrapped> opt = nullptr);

  /**
   * @brief     get begin iterator for the graph
   * @retval    const iterator
   */
  graph_const_iterator<SubGraphBase> cbegin() const {
    return graph.cbegin<SubGraphBase>();
  }

  /**
   * @brief     get end iterator for the graph
   * @retval    const iterator
   */
  graph_const_iterator<SubGraphBase> cend() const {
    return graph.cend<SubGraphBase>();
  }

  /**
   * @brief     get reverse begin iterator for the graph
   * @retval    const reverse iterator
   */
  graph_const_reverse_iterator<SubGraphBase> crbegin() const {
    return graph.crbegin<SubGraphBase>();
  }

  /**
   * @brief     get reverse end iterator for the graph
   * @retval    const reverse iterator
   */
  graph_const_reverse_iterator<SubGraphBase> crend() const {
    return graph.crend<SubGraphBase>();
  }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  graph_const_reverse_iterator<SubGraphBase> getBackwardingBeginIter() const {
    return crbegin();
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  graph_const_reverse_iterator<SubGraphBase> getBackwardingEndIter() const {
    return crend();
  }

  /**
   * @brief     getter of output dimension of graph
   * @retval    output tensor dim list
   */
  std::vector<TensorDim> getOutputDimension() const;

  /**
   * @brief     getter of input dimension of graph
   * @retval    input tensor dim list
   */
  std::vector<TensorDim> getInputDimension() const;

  /**
   * @brief Get the Batch Size object of current model
   *
   * @return unsigned int
   */
  unsigned int getBatchSize() const;

  /**
   * @brief     Copy the graph
   * @param[in] from Graph Object to copy
   * @retval    Graph Object copyed
   */
  NetworkGraph &copy(NetworkGraph &from) {
    graph.copy(from.graph);
    return *this;
  }

  /**
   * @brief initialize network graph
   *
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  int initialize(ExecutionMode mode = ExecutionMode::TRAIN,
                 const std::vector<Connection> &model_input_names = {},
                 const std::vector<Connection> &model_label_names = {});

  /**
   * @brief reinitialize network graph
   *
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  int reinitialize(const std::vector<Connection> &model_input_names = {},
                   const std::vector<Connection> &model_label_names = {});

  /**
   * @brief Create run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   */
  std::vector<Var_Grad *>
  finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                  const std::vector<Var_Grad *> &prev_inputs);

  /**
   * @brief Recreate run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   */
  std::vector<Var_Grad *>
  refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                    const std::vector<Var_Grad *> &prev_inputs);

  /** Interface for manager */

  /**
   * @brief Allocate memory for all the managed tensors
   *
   * @param[in] training If true, initialize derivates/gradients, else, do not.
   */
  void allocateTensors(ExecutionMode exec_mode_);

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocateTensors(bool dealloc_weights = false) {
    tensor_manager->deallocateTensors(dealloc_weights);
  }

  /**
   * @brief Allocate memory for all the managed weights
   */
  void allocateWeights(bool init = true) {
    for (auto it = cbegin(); it != cend(); ++it)
      return it->allocateWeights(init);
  }

  /**
   * @brief Deallocate memory for all the weights
   */
  void deallocateWeights() { tensor_manager->deallocateWeights(); }

  /**
   * @brief     Enable the memory optimizations for the network
   *
   * @param val true to enable, else false
   */
  void setMemoryOptimizations(bool val) {
    for (auto it = cbegin(); it != cend(); ++it)
      return it->setMemoryOptimizations(val);
  }

  /**
   * @brief     Create optimizer variable for every weights
   *
   * @param cb  Call back function which will return vector of dimension
   * @param request_only_trainable true when only request trainable weight
   */
  void requestOptimizerVariable(
    std::function<std::vector<TensorDim>(const TensorDim &)> cb,
    bool request_only_trainable = true);

  /**
   * @brief Feed inputs and labels to the graph
   *
   * @param inputs Input data
   * @param labels Label data
   */
  void setInputsLabels(const std::vector<Tensor> &inputs,
                       const std::vector<Tensor> &labels);

  /**
   * @brief Feed inputs and labels to the graph
   *
   * @param inputs Input data
   * @param labels Label data
   */
  void setInputsLabels(sharedConstTensors &inputs, sharedConstTensors &labels);

  /**
   * @brief Get the Output Tensors list for the graph
   *
   * @return std::vector<Tensor> List of output tensors
   * @note this tensor list is analogous to the label list
   */
  std::vector<Tensor> getOutputTensors() const;

  /**
   * @brief return model tensor type
   *
   * @return TensorDim::Format NCHW or NHWC
   */
  std::array<std::string, 3> getTensorType() {
    /**
     * @note This code written based on the assumption that he graph consists
     * with only one default subgraph node. If subgraphs have different
     * TensorType, then it needs to be update this function.
     */
    return (*cbegin())->getTensorType();
  };

  /**
   * @brief Flush data to the device
   *
   */
  void flushCache();

  /**
   * @brief Flush data to the device except order
   *
   * @param order except execution order
   */
  void flushCacheExcept(const unsigned int order);

  /**
   * @brief Load data of order to the device
   *
   * @param order execution order
   */
  void LoadTensors(const unsigned int order,
                   unsigned int remainder_lookahead = 0);

  /**
   * @brief check data of order is loaded
   *
   * @param order execution order
   */
  bool checkLoadComplete(const unsigned int order);

  /**
   * @brief check data of order is Unloaded
   *
   * @param order execution order
   */
  bool checkUnloadComplete(const unsigned int order);

  /**
   * @brief Load data of order to the device
   *
   * @param order execution order
   */
  void UnloadTensors(const unsigned int order);

#ifdef ENABLE_TEST
  /**
   * @brief Get layer node's tenexecution orders
   *
   * @param lnode layer node
   * @note this is for test purpose only
   */
  std::map<std::string, std::vector<unsigned int>>
  getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode);
#endif // ENABLE_TEST

  /**
   * @brief     reset the loss scale
   * @param[in] scale
   */
  void resetLossScale(float scale);

  /**
   * @brief     check if it is mixed precision training
   */
  bool isMixedPrecision() { return (!istrequal(tensor_dtype[1], "FP32")); }

  /**
   * @brief Get Number of Loaded WeightPool Tensor
   *
   * @return Number of Loaded WeightPool Tensor
   */
  unsigned int getNumLoadedWeightPoolTensors();

  /**
   * @brief Get Number of Loaded TensorPool Tensor
   *
   * @return Number of Loaded TensorPool Tensor
   */
  unsigned int getNumLoadedTensorPoolTensors();

private:
  std::map<std::string, std::string> sub_in_out; /** This is map to identify
                   input and output layer name of subgraph */
  std::shared_ptr<Manager> tensor_manager;       /**< tensors manager */

  GraphCore graph; /** core graph object consisting with SubGraphBase nodes */
  bool compiled;   /**< if the model graph is compiled */
  unsigned int batch_size;     /**< current batch_size */
  unsigned int graph_exec_end; /**< Inclusive, last execution order of the
                                  given graph */
  LayerNode
    *backward_iter_end;        /**< inclusive end node of the valid backward
                                  execution when initialized, nodes after this node
                                  does not required backwarding thus making it noop */
  LayerNode *forward_iter_end; /**< inclusive end node of the forward execution
                                  when initialize */

  /// @note *_list and *_dims must be synced at all times. Consider put it as a
  /// structure
  std::vector<std::string> label_list;  /**< identifier for the model labels */
  std::vector<std::string> input_list;  /**< identifier for the model inputs */
  std::vector<std::string> output_list; /**< identifier for the model outputs */
  std::vector<TensorDim> label_dims;    /**< graph label dimensions */
  std::vector<TensorDim> input_dims;    /**< graph input dimensions */

  bool optimize_memory;    /**< optimize memory */
  ExecutionMode exec_mode; /**< execution mode with which the graph has been
                              currently set or previously set */

  std::string tensor_format; /**< Model Tensor Format: NCHW or NHWC */
  std::string tensor_dtype_str;

  std::vector<std::string> tensor_dtype; /**< Model Tensor Type: FP32, FP16 */

  std::unordered_map<std::string, int>
    profile_keys; /**< profile keys based on the layer type */
  std::vector<Weight *>
    lazy_weights; /**< weights with delayed grad update, e.g., gradient
                     clipping, loss scaling */
  bool is_clip_grad;
  float loss_scale;
  unsigned int nan_count;
  unsigned int lookahead;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
