// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   manager.cpp
 * @date   2 Dec 2020
 * @brief  This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifdef __ANDROID__
#include <android/sharedmem.h>
#endif

#ifdef DEBUG
#include <cassert>
#endif

#include <fcntl.h>
#include <functional>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <activation_layer.h>
#include <basic_planner.h>
#include <bn_layer.h>
#include <layer_node.h>
#include <manager.h>
#include <nntrainer_log.h>
#include <optimized_v1_planner.h>
#include <util_func.h>

namespace nntrainer {
MMapedMemory::MMapedMemory(size_t size, bool allocate_fd_) :
  fd(-1),
  buf(nullptr),
  buf_size(0),
  allocate_fd(allocate_fd_) {

#ifndef __ANDROID__
  if (allocate_fd) {
    /// @todo create a file in tmpfs and bind to memfs
    /// memfd_create is not available for number of platforms so this is
    /// commented
    // auto fd_ = memfd_create("", 0);
    // if (fd_ < 0) {
    //   throw std::runtime_error("[Manager] creating mem fd failed");
    // }
    // if (ftruncate(fd_, size) < 0) {
    //   throw std::runtime_error("[Manager] truncating fd failed");
    // }
    ml_logi("[MMapedMemory] fd creation is not supported in this platform");
    allocate_fd = false;
  }
#endif
  int fd_ = -1;
  void *buf_ = nullptr;

  if (allocate_fd) {
#ifdef __ANDROID__
    /// unfortunately, memfd_create is not supported before android level 30
    fd_ = ASharedMemory_create("", size);
    if (fd_ < 0) {
      throw std::runtime_error("[MMapedMemory] creating mem fd failed");
    }

    if (ASharedMemory_setProt(fd_, PROT_READ | PROT_WRITE) < 0) {
      // unlink / close the given fd here
      close(fd_);
      throw std::runtime_error("[MMapedMemory] Setting prot failed");
    }

    buf_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
#endif
  } else {
    buf_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                fd_, 0);
  }

  if (buf_ == MAP_FAILED) {
#ifdef __ANDROID__
    if (fd_ != -1) {
      // unlink / close the given fd here
      close(fd_);
    }
#endif

    throw std::runtime_error("[MMapedMemory] mmap failed");
  }

  fd = fd_;
  buf = buf_;
  buf_size = size;

  ml_logd("[MMapedMemory] memory acquired size: %zu, fd: %d, addr: %p",
          buf_size, fd, buf);
}

MMapedMemory::~MMapedMemory() noexcept {
#ifdef DEBUG
  assert(buf_size > 0 && fd > 0);
#endif

  if (fd != -1) {
    if (close(fd) < 0) {
      ml_logw("[MMapedMemory] closing fd failed on destruction please check");
    }
  }

  if (buf != nullptr) {
    if (munmap(buf, buf_size) < 0) {
      ml_logw("[MMapedMemory] munmap failed on destruction please check");
    }
  }

  /// keeping the invariant although this is not necessary as of now
  fd = -1;
  buf = nullptr;
  buf_size = 0;
  ml_logd("[MMapedMemory] buf released");
}

void Manager::allocateWeights(unsigned int max_exec_order_) {
  if (!weight_pool.isAllocated()) {
    finalizeTensorPool(weight_pool, 0, max_exec_order_);
    weight_pool.allocate();
  }
}

void Manager::deallocateWeights() { weight_pool.deallocate(); }

/**
 * @brief Allocate memory for all the managed tensors
 */
void Manager::allocateTensors(unsigned int max_exec_order_) {
  allocateWeights(max_exec_order_);

  if (!tensor_pool.isAllocated()) {
    finalizeTensorPool(tensor_pool, 0, max_exec_order_);
    if (tensor_pool.minMemoryRequirement() > 0)
      tensor_pool.allocate();
  }
}

/**
 * @brief Deallocate memory for all the managed tensors
 */
void Manager::deallocateTensors(bool dealloc_weights) {
  if (dealloc_weights)
    deallocateWeights();

  tensor_pool.deallocate();
}

#ifdef LAYER_V1
void Manager::initializeTensorsInference(unsigned int max_exec_order_) {
  /**
   * A single buffer (shared_inout) provides memory for inputs and outputs of a
   * layer. Further, the output of layer i shares memory with input with layer
   * i+1. So, each alternate layer allocates memory from either the start of the
   * buffer or the end of the buffer, and use_first_last tracks this
   *
   * @note Label for the last layer is not initialized in inference.
   * @note Input for the first layer is not initialized in inference.
   */
  // Initialize shared input/output memory for inference
  // @note Memory for label is not allocated here as inference doesnt has label
  if (enable_inference_inout_memory_opt)
    shared_inout = Tensor(TensorDim({max_shared_inout}), false);

  bool use_first_last = 0;
  for (unsigned int idx = 0; idx < in_outs.size(); idx++) {
    auto &l_io = in_outs[idx];
    unsigned int offset = 0;
    bool is_first_layer = idx == 0;

    // For flatten layer, do not assign new memory
    if (idx > 0 && is_flat_type[idx])
      use_first_last = 1 - use_first_last;

    // In inference mode, do not allocate the memory for the input of the
    // first layer. These is the first entry in the in_outs. Inference() will
    // override input tensors of the first layer
    if (is_first_layer)
      continue;

    for (auto &io : l_io) {
      Tensor shared_inout_cur = Tensor();
      if (enable_inference_inout_memory_opt) {
        // if optimized
        if (use_first_last) {
          // Create tensor with from the front of shared tensor
          shared_inout_cur =
            shared_inout.getSharedDataTensor(io->getDim(), offset);
        } else {
          // Create tensor with from the back of shared tensor
          shared_inout_cur = shared_inout.getSharedDataTensor(
            io->getDim(),
            max_shared_inout - io->getDim().getDataLen() - offset);
        }
        offset += io->getDim().getDataLen();
      }
      io->initialize(shared_inout_cur, Tensor(), false);
    }
    use_first_last = 1 - use_first_last;
  }
}

void Manager::initializeTensorsTrain(unsigned int max_exec_order_) {
  // Initialize gradients
  initializeGradients();

  // Initialize shared derivative memory
  if (max_derivative_size > 0 && enable_activation_memory_opt)
    shared_deriv = Tensor(TensorDim({max_derivative_size}), false);
  for (unsigned int idx = 0; idx < in_outs.size(); idx++) {
    auto &l_io = in_outs[idx];
    unsigned int offset = 0;
    bool is_last_layer = idx == in_outs.size() - 1;

    for (auto &io : l_io) {
      // Last layer requires separate memory allocations for output and label
      // (deriv)
      if (enable_derivative_memory_opt && !is_last_layer) {
        // Training Mode with optimizations
        if (enable_activation_memory_opt &&
            (is_rnn_type[idx] || is_act_type[idx])) {
          io->initialize(
            Tensor(), shared_deriv.getSharedDataTensor(io->getDim(), offset));
          offset += io->getDim().getDataLen();
        } else {
          io->initializeShared();
        }

      } else {
        // Training Mode without optimizations
        io->initialize(Tensor(), Tensor(), true);
      }
    }
  }
}
#endif

/**
 * @brief     Create weights with the given spec
 *
 */
std::vector<Weight *> Manager::requestWeights(
  const GraphNode &node, const std::vector<Weight::Spec> &weights_spec,
  bool trainable, const std::vector<std::string> &shared_names) {
  const auto [forwarding_order, calcGradient_order, calcDerivative_order] =
    node.getExecutionOrder();
  std::vector<unsigned int> var_exec_order(
    {forwarding_order, calcGradient_order, calcDerivative_order});
  std::vector<unsigned int> grad_exec_order(
    {calcGradient_order, calcDerivative_order});

  TensorLifespan var_ls = TensorLifespan::MAX_LIFESPAN;
  TensorLifespan grad_ls = TensorLifespan::BACKWARD_FUNC_LIFESPAN;

  std::vector<Weight *> ret;
  size_t current_size = weights_v2.size();

  for (unsigned int i = 0; i < weights_spec.size(); ++i) {
    auto &[dim, t_initializer, w_reg, w_reg_const, need_gradient, name] =
      weights_spec.at(i);

    Tensor *var = nullptr, *grad = nullptr;
    bool is_dependent = !shared_names.empty();
    if (is_dependent) {
      const auto &shared_name = shared_names.at(i);
      /** case when shared names are given */
      var = weight_pool.requestPrerequestedTensor(
        dim, var_exec_order, var_ls,
        name,         /// name
        shared_name,  /// shared name
        t_initializer /// tensor initializer
      );

      if (trainable && need_gradient) {
        grad = tensor_pool.requestPrerequestedTensor(
          dim, grad_exec_order, grad_ls,
          name + Var_Grad::grad_suffix,        /// name
          shared_name + Var_Grad::grad_suffix, /// shared name
          Tensor::Initializer::ZEROS           /// tensor initializer
        );
      }

    } else {
      /** case requesting fresh weights */
      var = weight_pool.requestTensor(dim, var_exec_order, var_ls, name,
                                      t_initializer);

      if (trainable && need_gradient)
        grad = tensor_pool.requestTensor(dim, grad_exec_order, grad_ls,
                                         name + Var_Grad::grad_suffix,
                                         Tensor::Initializer::ZEROS);
    }

    weights_v2.emplace_back(
      std::make_unique<Weight>(var, grad, w_reg, w_reg_const, is_dependent));
  }

  std::transform(weights_v2.begin() + current_size, weights_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });

  return ret;
}

/**
 * @brief     Create weights with the given spec
 *
 */
std::vector<Var_Grad *>
Manager::requestTensors(const GraphNode &node,
                        const std::vector<Var_Grad::Spec> &tensors_spec,
                        const std::vector<std::string> &shared_names) {
  const auto [forwarding_order, calcGradient_order, calcDerivative_order] =
    node.getExecutionOrder();

  std::vector<Var_Grad *> ret;
  size_t current_size = tensors_v2.size();

  for (unsigned int i = 0; i < tensors_spec.size(); ++i) {
    auto const &[dim, t_init, need_grad, name, tspan] = tensors_spec.at(i);

    std::vector<unsigned int> var_exec_order;
    std::vector<unsigned int> grad_exec_order;

    /** usage for tensors */
    if (enum_class_logical_and<TensorLifespan>(
          tspan, TensorLifespan::FORWARD_FUNC_LIFESPAN))
      var_exec_order.push_back(forwarding_order);

    /** usage for tensors gradient in backwarding */
    if (enum_class_logical_and<TensorLifespan>(
          tspan, TensorLifespan::BACKWARD_FUNC_LIFESPAN)) {
      var_exec_order.push_back(calcGradient_order);
      grad_exec_order.push_back(calcGradient_order);

      var_exec_order.push_back(calcDerivative_order);
      grad_exec_order.push_back(calcDerivative_order);
    }

    bool is_dependent = !shared_names.empty();
    Tensor *var = nullptr, *grad = nullptr;

    if (is_dependent) {
      [[maybe_unused]] const auto &shared_name = shared_names.at(i);
      var = tensor_pool.requestPrerequestedTensor(dim, var_exec_order, tspan,
                                                  name, shared_name, t_init);
      if (need_grad && tspan > TensorLifespan::FORWARD_FUNC_LIFESPAN) {
        grad = tensor_pool.requestPrerequestedTensor(
          dim, grad_exec_order, tspan,
          name + Var_Grad::grad_suffix, /// name
          shared_name + Var_Grad::grad_suffix,
          Tensor::Initializer::ZEROS /// tensor initializer
        );
      }

    } else {
      var = tensor_pool.requestTensor(dim, var_exec_order, tspan, name, t_init);

      if (need_grad && tspan > TensorLifespan::FORWARD_FUNC_LIFESPAN) {
        grad = tensor_pool.requestTensor(
          dim, grad_exec_order, tspan,
          name + Var_Grad::grad_suffix, /// name
          Tensor::Initializer::ZEROS    /// tensor initializer
        );
      }
    }

    tensors_v2.emplace_back(std::make_unique<Var_Grad>(var, grad));
  }

  std::transform(tensors_v2.begin() + current_size, tensors_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });

  return ret;
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestInputs(const GraphNode &node,
                       const std::vector<TensorDim> &inputs_dim,
                       const std::vector<std::string> &outputs_name) {
  const auto [forwarding_order, calcGradient_order, calcDerivative_order] =
    node.getExecutionOrder();
  std::vector<unsigned int> var_exec_order(
    {forwarding_order, calcGradient_order});
  std::vector<unsigned int> grad_exec_order({calcDerivative_order});

  /** batch normalization layer uses input in forwarding only */
  if (node.getType() == BatchNormalizationLayer::type)
    var_exec_order = {forwarding_order};

  TensorLifespan var_ls = TensorLifespan::ITERATION_LIFESPAN;
  TensorLifespan grad_ls = TensorLifespan::ITERATION_LIFESPAN;

  std::vector<Var_Grad *> ret;
  size_t current_size = inputs_v2.size();

  for (unsigned int idx = 0; idx < inputs_dim.size(); idx++) {
    auto const &dim = inputs_dim[idx];
    Tensor *var = nullptr, *grad = nullptr;
    const std::string &var_name =
      node.getName() + std::string(":input") + std::to_string(idx);
    if (!outputs_name.empty()) {
      var = tensor_pool.requestPrerequestedTensor(
        dim, /// tensor dim
        var_exec_order, var_ls,
        var_name,                 /// name
        outputs_name[idx],        /// shared name
        Tensor::Initializer::NONE /// tensor initializer
      );

      grad = tensor_pool.requestPrerequestedTensor(
        dim, /// tensor dim
        grad_exec_order, grad_ls,
        var_name + Var_Grad::grad_suffix,          /// name
        outputs_name[idx] + Var_Grad::grad_suffix, /// shared name
        Tensor::Initializer::ZEROS                 /// tensor initializer
      );
    } else if (!node.getInputConnections().empty()) {
      var = tensor_pool.requestTensor(
        dim, /// tensor dim
        var_exec_order, var_ls,
        var_name,                 /// name
        Tensor::Initializer::NONE /// tensor initializer
      );

      grad = tensor_pool.requestTensor(
        dim, /// tensor dim
        grad_exec_order, grad_ls,
        var_name + Var_Grad::grad_suffix, /// name
        Tensor::Initializer::ZEROS        /// tensor initializer
      );
    } else {
      /** requesting externally allocated tensor for input */
      var = tensor_pool.requestExternallyAllocateTensor(
        dim,                      /// tensor dim
        var_name,                 /// name
        Tensor::Initializer::NONE /// tensor initializer
      );

#ifdef ENABLE_TEST
      grad = tensor_pool.requestTensor(
        dim, /// tensor dim
        grad_exec_order, grad_ls,
        var_name + Var_Grad::grad_suffix, /// name
        Tensor::Initializer::ZEROS        /// tensor initializer
      );
#else
      grad = tensor_pool.requestExternallyAllocateTensor(
        dim,                              /// tensor dim
        var_name + Var_Grad::grad_suffix, /// name
        Tensor::Initializer::ZEROS        /// tensor initializer
      );
#endif
    }

    inputs_v2.emplace_back(std::make_unique<Var_Grad>(var, grad));
  }

  ret.reserve(inputs_dim.size());
  std::transform(inputs_v2.begin() + current_size, inputs_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });

  return ret;
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestOutputs(const GraphNode &node,
                        const std::vector<TensorDim> &outputs_dim,
                        const std::vector<std::string> &inputs_name) {
  const auto [forwarding_order, calcGradient_order, calcDerivative_order] =
    node.getExecutionOrder();
  std::vector<unsigned int> var_exec_order({forwarding_order});
  if (node.getType() == ActivationLayer::type)
    /** TODO: if removing this reduces memory consumption, resolve this */
    var_exec_order.push_back(calcDerivative_order);
  std::vector<unsigned int> grad_exec_order(
    {calcGradient_order, calcDerivative_order});

  TensorLifespan var_ls = TensorLifespan::ITERATION_LIFESPAN;
  TensorLifespan grad_ls = TensorLifespan::ITERATION_LIFESPAN;

  std::vector<Var_Grad *> ret;
  size_t current_size = outputs_v2.size();

  for (unsigned int idx = 0; idx < outputs_dim.size(); idx++) {
    auto const &dim = outputs_dim[idx];
    Tensor *var = nullptr, *grad = nullptr;
    const std::string &var_name =
      node.getName() + std::string(":output") + std::to_string(idx);
    if (!inputs_name.empty()) {
      var = tensor_pool.requestPrerequestedTensor(
        dim, /// tensor dim
        var_exec_order, var_ls, var_name,
        inputs_name[idx],         /// name
        Tensor::Initializer::NONE /// tensor initializer
      );

      /** skip requesting tensor for label */
      if (!node.getOutputConnections().empty()) {
        grad = tensor_pool.requestPrerequestedTensor(
          dim, /// tensor dim
          grad_exec_order, grad_ls,
          var_name + Var_Grad::grad_suffix,         /// name
          inputs_name[idx] + Var_Grad::grad_suffix, /// shared name
          Tensor::Initializer::ZEROS                /// tensor initializer
        );
      } else {
        /** requesting externally allocated tensor for label */
        grad = tensor_pool.requestExternallyAllocateTensor(
          dim,                              /// tensor dim
          var_name + Var_Grad::grad_suffix, /// name
          Tensor::Initializer::ZEROS        /// tensor initializer
        );
      }
    } else {
      var = tensor_pool.requestTensor(
        dim, /// tensor dim
        var_exec_order, var_ls,
        var_name,                 /// name
        Tensor::Initializer::NONE /// tensor initializer
      );

      if (!node.getOutputConnections().empty()) {
        grad = tensor_pool.requestTensor(
          dim, /// tensor dim
          grad_exec_order, grad_ls,
          var_name + Var_Grad::grad_suffix, /// name
          Tensor::Initializer::ZEROS        /// tensor initializer
        );
      } else {
        /** requesting externally allocated tensor for label */
        grad = tensor_pool.requestExternallyAllocateTensor(
          dim,                              /// tensor dim
          var_name + Var_Grad::grad_suffix, /// name
          Tensor::Initializer::ZEROS        /// tensor initializer
        );
      }
    }

    outputs_v2.emplace_back(std::make_unique<Var_Grad>(var, grad));
  }

  std::transform(outputs_v2.begin() + current_size, outputs_v2.end(),
                 std::back_inserter(ret),
                 [](auto const &elem) { return elem.get(); });

  return ret;
}

std::pair<unsigned int, unsigned int>
Manager::getMinMaxTensorExecutionOrder(const std::string &name) {
  auto orders = tensor_pool.getExecutionOrder(name);
  auto [min_, max_] = std::minmax_element(orders.begin(), orders.end());
  return {*min_, *max_};
}

bool Manager::isFirstAccess(const std::string &name,
                            unsigned current_execution) {
  /// @todo add cache machanism, eg) sort at finalizing requesting
  return getMinMaxTensorExecutionOrder(name).first == current_execution;
}

bool Manager::isLastAccess(const std::string &name,
                           unsigned current_execution) {
  /// @todo add cache machanism, eg) sort at finalizing requesting
  return getMinMaxTensorExecutionOrder(name).second == current_execution;
}

/**
 * @brief     Create tensors with the given spec
 *
 */
std::vector<Tensor *> Manager::requestWeightOptimizerVariables(
  const std::vector<TensorDim> &dims, const std::string &name,
  const TensorLifespan &lifespan, Tensor::Initializer initializer) {
  auto const &exec_order = weight_pool.getExecutionOrder(name);

  std::vector<Tensor *> ret;
  ret.reserve(dims.size());

  for (unsigned int idx = 0; idx < dims.size(); idx++)
    ret.push_back(tensor_pool.requestTensor(dims[idx], exec_order, lifespan,
                                            name + ":opt" + std::to_string(idx),
                                            initializer));

  return ret;
}

std::vector<Weight *> Manager::getWeights() {
  std::vector<Weight *> all_weights;

  for (auto &w : weights_v2) {
    all_weights.push_back(w.get());
  }

  return all_weights;
}

void Manager::finalizeTensorPool(TensorPool &pool, unsigned int start,
                                 unsigned int end) {
  if (enable_optimizations)
    pool.finalize(OptimizedV1Planner(), start, end);
  else
    pool.finalize(BasicPlanner(), start, end);
}

} // namespace nntrainer
