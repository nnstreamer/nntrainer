// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
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

#include <execution_mode.h>
#include <graph_core.h>
#include <layer_node.h>
#include <manager.h>

namespace nntrainer {

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
    tensor_dtype(split("FP32-FP32", std::regex("\\-"))) {}

  /**
   * @brief     Constructor of NeuralNetwork Graph Class
   * @param[in] enable_swap enable memory swap for tensor
   * @param[in] swap_path memory swap file path when the swap is enabled
   */
  NetworkGraph(bool enable_swap, const std::string &swap_path = "",
               unsigned int lookahead = 0,
               const std::string &tensor_format_ = "NCHW",
               const std::string &tensor_dtype_ = "FP32-FP32") :
    tensor_manager(std::make_shared<Manager>(enable_swap, swap_path, lookahead,
                                             tensor_format_, tensor_dtype_)),
    graph(),
    compiled(false),
    batch_size(0),
    graph_exec_end(0),
    backward_iter_end(nullptr),
    forward_iter_end(nullptr),
    optimize_memory(true),
    exec_mode(ExecutionMode::TRAIN),
    tensor_format(tensor_format_),
    tensor_dtype(split(tensor_dtype_, std::regex("\\-"))) {}

  /**
   * @brief   Destructor of the NeuralNetwork Graph class
   *
   */
  ~NetworkGraph() = default;

  /**
   * @brief     Compile the graph
   * @param[in] loss_type loss for the graph
   * returns ML_ERROR_NONE on success, error on failure
   */
  int compile(const std::string &loss_type);

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
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int size() const { return graph.size(); }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const { return graph.empty(); }

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
    return std::static_pointer_cast<LayerNode>(graph.getSortedNode(ith));
  }

  /**
   * @brief getter of LayerNode with layer name
   * @param[in] layer name
   * @retval LayerNode
   */
  std::shared_ptr<LayerNode> getLayerNode(const std::string &layer_name) const {
    return std::static_pointer_cast<LayerNode>(graph.getNode(layer_name));
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
   * @param apply_func apply function
   */
  static void applyGradients(LayerNode *node,
                             const std::function<void(Weight &)> &apply_func);

  /**
   * @brief     forwarding network graph
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors forwarding(
    bool training = false,
    std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
      [](std::shared_ptr<LayerNode>, bool) {},
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr);

  /**
   * @brief     backwarding the network graph
   * @param[in] iteration current iteration number
   * @param[in] backwarding_op operation for the backwarding
   * @param[in] apply_grad_clip_op operation for applying the clip gradients
   */
  void backwarding(
    int iteration,
    std::function<void(std::shared_ptr<LayerNode>, int)> &backwarding_op,
    std::function<void(Weight &, int)> &apply_grad_clip_op,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr) const;

  /**
   * @brief     get begin iterator for the graph
   * @retval    const iterator
   */
  graph_const_iterator<LayerNode> cbegin() const {
    return graph.cbegin<LayerNode>();
  }

  /**
   * @brief     get end iterator for the graph
   * @retval    const iterator
   */
  graph_const_iterator<LayerNode> cend() const {
    return graph.cend<LayerNode>();
  }

  /**
   * @brief     get reverse begin iterator for the graph
   * @retval    const reverse iterator
   */
  graph_const_reverse_iterator<LayerNode> crbegin() const {
    return graph.crbegin<LayerNode>();
  }

  /**
   * @brief     get reverse end iterator for the graph
   * @retval    const reverse iterator
   */
  graph_const_reverse_iterator<LayerNode> crend() const {
    return graph.crend<LayerNode>();
  }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  graph_const_reverse_iterator<LayerNode> getBackwardingBeginIter() const {
    return crbegin();
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  graph_const_reverse_iterator<LayerNode> getBackwardingEndIter() const {
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
  int initialize(const std::vector<Connection> &model_input_names = {},
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
  void allocateWeights() {
    tensor_manager->allocateWeights(
      std::get<3>(backward_iter_end->getExecutionOrder()));
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
    tensor_manager->setOptimizations(val);
    optimize_memory = val;
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
  std::array<const std::string, 3> getTensorType() {

    return {tensor_format, tensor_dtype[0], tensor_dtype[1]};
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

private:
  std::map<std::string, std::string> sub_in_out; /** This is map to identify
                   input and output layer name of subgraph */
  std::shared_ptr<Manager> tensor_manager;       /**< tensors manager */

  GraphCore graph;             /** core graph object */
  bool compiled;               /**< if the model graph is compiled */
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

  std::vector<std::string> tensor_dtype; /**< Model Tensor Type: FP32, FP16 */

  std::unordered_map<std::string, int>
    profile_keys; /**< profile keys based on the layer type */
  std::vector<Weight *>
    clip_weights; /**< weights with global norm based clipping enabled */

  /**
   * @brief     topological sort
   * @param[in] ith index of LayerNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void topologicalSortUtil(unsigned int ith, std::vector<bool> &visited,
                           std::stack<std::shared_ptr<LayerNode>> &Stack);

  /**
   * @brief     check if graph is ready to compile.
   * @retval #ML_ERROR_NONE graph is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int isCompilable();

  /**
   * @brief     check if the compiled graph is of correct form.
   * @retval #ML_ERROR_NONE graph is compiled correctly
   * @retval #ML_ERROR_INVALID_PARAMETER did not compile correctly
   */
  int checkCompiledGraph();

  /**
   * @brief     mark nodes required for backwarding.
   */
  void markNodesForBackwarding();

  /**
   * @brief     adding loss layer at last position
   * @param[in] loss_type loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLossLayer(const std::string &loss_type);

  /**
   * @brief     set output connections for all the layers
   */
  void setOutputConnections();

  /**
   * @brief     Ensure that layer has a name.
   * @param[in] layer Layer whose name is to be ensured to be valid
   * @param[in] prefix Prefix to be attached to the layer name
   * @param[in] postfix Postfix to be attached to the layer name
   * @param[in] force_rename If the layer must be forcefully rename
   * @details   Ensures that the layer has a unique and a valid name. A valid
   * name pre-assigned to the layer can be changed if force_rename is enabled.
   */
  void ensureName(std::shared_ptr<Layer> layer, const std::string &prefix = "",
                  const std::string &postfix = "", bool force_rename = false);

  /**
   * @brief Create new LayerNode and add into Graph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayerNode(std::unique_ptr<Layer> layer);

  /**
   * @brief finalize already added loss layers
   *
   * @details This involves verify if the requirements of the added loss layers
   * match and merging loss layers with activation layers if needed.
   */
  void finalizeLossLayer();

  /**
   * @brief Set the order of execution for all the nodes in the graph
   *
   * @details This sets the order of execution using the order from the
   * topological sort. The order of forwarding matches the topological sort. The
   * order for backwarding is in the exact reverse order. The calcDerivative()
   * is expected to be called right after calcGradient().
   */
  void setExecutionOrder();

  /**
   * @brief Set external data to the given tensors with name
   *
   * @param data External data
   * @param names Names of the tensor to set the data to
   */
  void setExternalTensors(const std::vector<Tensor> &data,
                          const std::vector<std::string> names);

  /**
   * @brief     Optimize the graph memory utilization for in-place operations
   */
  void inPlaceOptimize();

  /**
   * @brief     Check if the given node can execute in-place
   *
   * @param lnode node to check for in-place execution
   *
   * @return the mode of inplace for the layer
   */
  InPlace canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode);

  /**
   * @brief compute optimized backward end. This function calculated the valid
   * end of the graph backward, if memory_optimize is unset, this returns
   * beginning of the graph node.
   *
   * @return end of the backward iter;
   */
  LayerNode *computeBackwardEnd();
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
