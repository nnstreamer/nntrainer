// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    subgraph_base.h
 * @date    07 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is a Network SubGraph Class for Neural Network
 *
 */

#ifndef __SUBGRAPH_BASE_H__
#define __SUBGRAPH_BASE_H__
#ifdef __cplusplus

#include <list>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include <graph_core.h>
#include <layer_node.h>
#include <manager.h>
#include <optimizer_wrapped.h>

namespace nntrainer {

using ExecutionMode = ml::train::ExecutionMode;

class Connection;
/**
 * @class   NeuralNetwork SubGraph Class
 * @brief   NeuralNetwork SubGraph Class which manage layers
 */
class SubGraphBase {

public:
  /**
   * @brief     Constructor of NeuralNetwork SubGraph Class
   */
  SubGraphBase(std::shared_ptr<Manager> tm, unsigned int lookahead_ = 0) :
    tensor_manager(tm),
    subgraph(),
    compiled(false),
    batch_size(0),
    graph_exec_end(0),
    backward_iter_end(nullptr),
    forward_iter_end(nullptr),
    optimize_memory(true),
    exec_mode(ExecutionMode::TRAIN),
    tensor_format("NCHW"),
    tensor_dtype(split("FP32-FP32", getRegex("\\-"))),
    lookahead(lookahead_) {
    nan_count = 0;
  }

  /**
   * @brief   Destructor of the NeuralNetwork SubGraph class
   *
   */
  ~SubGraphBase() = default;

  /**
   * @brief     Compile the subgraph
   * @param[in] loss_type loss for the subgraph
   * returns ML_ERROR_NONE on success, error on failure
   */
  int compile(const std::string &loss_type);

  /**
   * @brief Create new LayerNode and add into SubGraph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayer(std::shared_ptr<LayerNode> layer);

  /**
   * @brief get current flat subgraph from the model before sorting
   * @note subgraph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current flat subgraph
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
  unsigned int size() const { return subgraph.size(); }

  /**
   * @brief get if the subgraph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const { return subgraph.empty(); }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(SubGraphBase &lhs, SubGraphBase &rhs) {
    /// @fixme this swap function need maintenance
    using std::swap;

    swap(lhs.subgraph, rhs.subgraph);
  }

  /**
   * @brief getter of Sorted LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  std::shared_ptr<LayerNode> getSortedLayerNode(unsigned int ith) const {
    return std::static_pointer_cast<LayerNode>(subgraph.getSortedNode(ith));
  }

  /**
   * @brief getter of LayerNode with layer name
   * @param[in] layer name
   * @retval LayerNode
   */
  std::shared_ptr<LayerNode> getLayerNode(const std::string &layer_name) const {
    return std::static_pointer_cast<LayerNode>(subgraph.getNode(layer_name));
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
   * @brief     forwarding network subgraph
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors forwarding(
    bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool swap_mode = false);

  /**
   * @brief     forwarding network subgraph
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
   * @brief     backwarding the network subgraph
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
   * @brief     get begin iterator for the subgraph
   * @retval    const iterator
   */
  graph_const_iterator<LayerNode> cbegin() const {
    return subgraph.cbegin<LayerNode>();
  }

  /**
   * @brief     get end iterator for the subgraph
   * @retval    const iterator
   */
  graph_const_iterator<LayerNode> cend() const {
    return subgraph.cend<LayerNode>();
  }

  /**
   * @brief     get reverse begin iterator for the subgraph
   * @retval    const reverse iterator
   */
  graph_const_reverse_iterator<LayerNode> crbegin() const {
    return subgraph.crbegin<LayerNode>();
  }

  /**
   * @brief     get reverse end iterator for the subgraph
   * @retval    const reverse iterator
   */
  graph_const_reverse_iterator<LayerNode> crend() const {
    return subgraph.crend<LayerNode>();
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
   * @brief     getter of output dimension of subgraph
   * @retval    output tensor dim list
   */
  std::vector<TensorDim> getOutputDimension() const;

  /**
   * @brief     getter of input dimension of subgraph
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
   * @brief     Copy the subgraph
   * @param[in] from SubGraphBase Object to copy
   * @retval    SubGraph Object copyed
   */
  SubGraphBase &copy(SubGraphBase &from) {
    subgraph.copy(from.subgraph);
    return *this;
  }

  /**
   * @brief initialize network subgraph
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
   * @brief reinitialize network subgraph
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
    unsigned int max_exec_order =
      std::get<3>(backward_iter_end->getExecutionOrder());

    if (exec_mode == ExecutionMode::INFERENCE)
      max_exec_order = std::get<0>(forward_iter_end->getExecutionOrder());
    tensor_manager->allocateWeights(max_exec_order, init);
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
   * @brief Feed inputs and labels to the subgraph
   *
   * @param inputs Input data
   * @param labels Label data
   */
  void setInputsLabels(const std::vector<Tensor> &inputs,
                       const std::vector<Tensor> &labels);

  /**
   * @brief Feed inputs and labels to the subgraph
   *
   * @param inputs Input data
   * @param labels Label data
   */
  void setInputsLabels(sharedConstTensors &inputs, sharedConstTensors &labels);

  /**
   * @brief Get the Output Tensors list for the subgraph
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

  /**
   * @brief Load data of order to the device
   *
   * @param order execution order
   */
  void LoadTensors(const unsigned int order);

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

private:
  std::map<std::string, std::string> sub_in_out; /** This is map to identify
                   input and output layer name of subgraph */
  std::shared_ptr<Manager> tensor_manager;       /**< tensors manager */

  GraphCore subgraph;          /** core graph object */
  bool compiled;               /**< if the subgraph is compiled */
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
  std::vector<std::string>
    label_list; /**< identifier for the subgraph labels */
  std::vector<std::string>
    input_list; /**< identifier for the subgraph inputs */
  std::vector<std::string>
    output_list;                     /**< identifier for the subgraph outputs */
  std::vector<TensorDim> label_dims; /**< subgraph label dimensions */
  std::vector<TensorDim> input_dims; /**< subgraph input dimensions */

  bool optimize_memory;    /**< optimize memory */
  ExecutionMode exec_mode; /**< execution mode with which the subgraph has been
                              currently set or previously set */

  std::string tensor_format; /**< Model Tensor Format: NCHW or NHWC */

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

  /**
   * @brief     topological sort
   * @param[in] ith index of LayerNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void topologicalSortUtil(unsigned int ith, std::vector<bool> &visited,
                           std::stack<std::shared_ptr<LayerNode>> &Stack);

  /**
   * @brief     check if subgraph is ready to compile.
   * @retval #ML_ERROR_NONE subgraph is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int isCompilable();

  /**
   * @brief     check if the compiled subgraph is of correct form.
   * @retval #ML_ERROR_NONE subgraph is compiled correctly
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
   * @brief Create new LayerNode and add into SubGraph
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
   * @brief Set the order of execution for all the nodes in the subgraph
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
   * @brief     Optimize the subgraph memory utilization for in-place operations
   */
  void inPlaceOptimize();

  /**
   * @brief     Check if the given node can execute in-place
   *
   * @param lnode node to check for in-place execution
   *
   * @return the mode of inplace for the layer
   */
  InPlaceType canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode);

  /**
   * @brief compute optimized backward end. This function calculated the valid
   * end of the subgraph backward, if memory_optimize is unset, this returns
   * beginning of the subgraph node.
   *
   * @return end of the backward iter;
   */
  LayerNode *computeBackwardEnd();

  /**
   * @brief forwarding_op function
   */
  void forwarding_op(std::shared_ptr<LayerNode> node, bool training,
                     bool swap_mode = false);
  /**
   * @brief forwarding_op function
   */
  void incremental_forwarding_op(std::shared_ptr<LayerNode> node,
                                 unsigned int from, unsigned int to,
                                 bool training);
  /**
   * @brief backwarding_op
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
  bool backwarding_op(std::shared_ptr<LayerNode> node, int iteration,
                      std::function<bool(void *userData)> stop_cb,
                      void *user_data, bool is_grad_opt_mode,
                      std::shared_ptr<OptimizerWrapped> opt);
  /**
   * @brief backwarding_op
   * @param[in] iteration current iteration number
   * @param[in] opt shared ptr of the optimizer used for applyGradients (opt is
   *            passed from neuralnetwork)
   */
  void lazy_apply_grad_op(Weight &w, int iteration,
                          std::shared_ptr<OptimizerWrapped> opt);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBGRAPH_BASE_H__ */
