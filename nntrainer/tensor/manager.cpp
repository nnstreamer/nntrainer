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
#include <flatten_layer.h>
#include <layer_node.h>
#include <manager.h>
#include <nntrainer_log.h>
#include <rnn.h>
#include <util_func.h>

static constexpr bool LAYER_V2 = true;

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

Manager::Manager(bool enable_gradient_memory_opt_,
                 bool enable_derivative_memory_opt_,
                 bool enable_activation_memory_opt_,
                 bool enable_inference_inout_memory_opt_) :
  total_weight_size(0),
  total_grad_size(0),
  max_grad_size(0),
  max_derivative_size(0),
  max_shared_inout(0),
  weights_initialized(false),
  tensors_initialized(false),
  weights_allocated(false),
  tensors_allocated(false),
  model_training(false),
  enable_gradient_memory_opt(enable_gradient_memory_opt_),
  enable_derivative_memory_opt(enable_derivative_memory_opt_),
  enable_activation_memory_opt(enable_activation_memory_opt_),
  enable_inference_inout_memory_opt(enable_inference_inout_memory_opt_),
  use_shared_memory(false) {}

/**
 * @brief Destructor
 */
Manager::~Manager() { reset(); }

/**
 * @brief     Add weight to be tracked and updated with nntrainer
 */
void Manager::trackWeight(std::reference_wrapper<Weight> w) {
  /// @warning this does not track the weight size etcs.. This might break when
  /// use_shared_memory = true
  std::vector<std::reference_wrapper<Weight>> temp = {w};
  weights.emplace_back(temp);
  weights_initialized = false;
}

/**
 * @brief     Add weights to be tracked and updated with nntrainer
 */
void Manager::trackWeights(std::vector<Weight> &ws) {
  if (weights_initialized || tensors_initialized)
    throw std::runtime_error("Cannot track more weights after initialize.");

  std::vector<std::reference_wrapper<Weight>> layer_weights;
  layer_weights.reserve(ws.size());

  unsigned int weight_size = 0;
  unsigned int grad_size = 0;

  for (auto &w : ws) {
    layer_weights.emplace_back(std::ref(w));
    unsigned int len = w.getDim().getDataLen();
    weight_size += len;
    if (w.needsGradient())
      grad_size += len;
  }

  weights.push_back(layer_weights);

  total_weight_size += weight_size;
  total_grad_size += grad_size;
  max_grad_size = std::max(max_grad_size, grad_size);
  weights_initialized = false;
}

Manager::AllocFunc Manager::getAllocFunc(bool is_weight) {
  AllocFunc allocate_none = [](const TensorDim &dim, unsigned int) {
    return Tensor();
  };

  AllocFunc allocate_func = allocate_none;

  if (use_shared_memory) {
    /**< use_shared_memory has been deprecated */

    /// this creates memory and sets to @a memory and returns AllocFunc
    auto get_allocfunc =
      [allocate_none](const unsigned int weight_size,
                      std::unique_ptr<MMapedMemory> &memory) -> AllocFunc {
      if (weight_size == 0) {
        return allocate_none;
      }

      if (weight_size >=
          std::numeric_limits<unsigned int>::max() / sizeof(float)) {
        throw std::invalid_argument(
          "weights exceed maximum size supported for shared memory");
      }
      unsigned int byte_size = weight_size * sizeof(float);
      memory = std::make_unique<MMapedMemory>(byte_size, true);
      return [&memory, byte_size](const TensorDim &dim, size_t offset) {
        return Tensor::Map(memory->typedBuffer<float>(), byte_size, dim,
                           offset);
      };
    };

    if (is_weight) {
      /** For weights */
      allocate_func = get_allocfunc(total_weight_size, weight_mmaped_memory);
    } else {
      /** for gradients */
      unsigned int grad_size =
        enable_gradient_memory_opt ? max_grad_size : total_grad_size;
      allocate_func = get_allocfunc(grad_size, grad_mmaped_memory);
    }
  } else if (!is_weight) {
    /** only for gradients */
    if (max_grad_size > 0 && enable_gradient_memory_opt) {
      // create a lazily allocated shared_grad
      shared_grad = Tensor(TensorDim({max_grad_size}), false);

      allocate_func = [this](const TensorDim &dim, unsigned int offset) {
        return shared_grad.getSharedDataTensor(dim, offset);
      };
    }
  }

  return allocate_func;
}

std::pair<unsigned int, unsigned int>
Manager::getValidity(const std::string &name) {
  /** @todo calculate validity based on lifespan and usage */
  return {0, std::numeric_limits<unsigned>::max()};
}

/**
 * @brief Allocate and initialize the weight variable
 */
void Manager::initializeWeights() {

  if (weights_initialized)
    return;

  if (LAYER_V2) {
    for (auto &w : weights_v2) {
      w->initializeVariable();
      auto const &t_validity = getValidity(w->getName());
      tensor_token_map[w->getName()] = pool.requestMemory(
        w->getVariableRef().bytes(), t_validity.first, t_validity.second);
    }
    pool.planLayout(BasicPlanner());
  } else {
    if (total_weight_size == 0) {
      ml_logw(
        "Nothing done on initialize because there is no weight registered");
      return;
    }

    AllocFunc allocate_weight = getAllocFunc(true);

    unsigned int weight_offset = 0;
    for (auto &l_w : weights) {
      for (auto &w : l_w) {
        Weight &weight = w.get();
        auto dim = weight.getDim();

        Tensor weight_prealloc = allocate_weight(dim, weight_offset);
        weight_offset += dim.getDataLen();

        weight.initializeVariable(weight_prealloc);
      }
    }
  }

  weights_initialized = true;
}

void Manager::allocateWeights() {
  if (weights_allocated)
    return;

  if (LAYER_V2) {
    pool.allocate();
    for (auto &w : weights_v2) {
      w->getVariableRef().setData(
        pool.getMemory(tensor_token_map[w->getName()]), true);
    }
  } else {
    for (auto &l_w : weights) {
      for (auto &w : l_w) {
        Weight &weight = w.get();
        weight.allocateVariable();
      }
    }
  }

  weights_allocated = true;
}

void Manager::deallocateWeights() {
  if (LAYER_V2) {
    for (auto &w : weights_v2) {
      /** this just nullifies the set pointer to avoid access to released memory
       */
      w->deallocateVariable();
    }
    pool.deallocate();
  } else {
    for (auto &l_w : weights) {
      for (auto &w : l_w) {
        Weight &weight = w.get();
        weight.deallocateVariable();
      }
    }
  }

  weights_allocated = false;
}

void Manager::allocateGradients() {
  /** Allocate the source tensors for shared memories */
  if (!shared_grad.empty())
    shared_grad.allocate();

  if (LAYER_V2) {
    for (auto &w : weights_v2) {
      w->allocateGradient();
    }
  } else {
    for (auto &l_w : weights) {
      for (auto &w : l_w) {
        Weight &weight = w.get();
        weight.allocateGradient();
      }
    }
  }
}

void Manager::deallocateGradients() {
  shared_grad.deallocate();

  if (LAYER_V2) {
    for (auto &w : weights_v2) {
      w->deallocateGradient();
    }
  } else {
    for (auto &l_w : weights) {
      for (auto &w : l_w) {
        Weight &weight = w.get();
        weight.deallocateGradient();
      }
    }
  }
}

/**
 * @brief Initialize the weight gradients
 */
void Manager::initializeGradients() {
  if (LAYER_V2) {
    for (auto &w : weights_v2) {
      w->initializeGradient();
      // auto exec_order = tensor_exec_order[w.getName()];
      // tensor_map(&w->getGradientRef(), requestMemory(w.getDim().size(),
      //       std::get<1>(exec_order), std::get<2>(exec_order) + 1));
    }
  } else {
    if (total_weight_size == 0) {
      ml_logw(
        "Nothing done on initialize because there is no weight registered");
      return;
    }

    AllocFunc allocate_grad = getAllocFunc(false);

    unsigned int grad_offset = 0;
    for (auto &l_w : weights) {
      if (enable_gradient_memory_opt) {
        grad_offset = 0;
      }
      for (auto &w : l_w) {
        Weight &weight = w.get();
        auto dim = weight.getDim();
        Tensor grad_prealloc = Tensor();
        if (weight.needsGradient()) {
          grad_prealloc = allocate_grad(dim, grad_offset);
          grad_offset += dim.getDataLen();
        }
        weight.initializeGradient(grad_prealloc);
      }
    }
  }
}

/**
 * @brief Track the inputs/ouputs of the layer
 * still derivative memory needs to be allocated
 */
std::vector<std::shared_ptr<Var_Grad>> &
Manager::trackLayerInOuts(const std::string &layer_type,
                          const std::string &layer_name,
                          const std::vector<TensorDim> &inout_dim) {
  if (tensors_initialized)
    throw std::runtime_error(
      "Cannot track more inputs/outputs after initialize.");

  int cnt = 0;
  bool is_act_layer = layer_type == ActivationLayer::type;
  bool is_flat_layer = layer_type == FlattenLayer::type;
  bool is_rnn_layer = layer_type == RNNLayer::type;

  unsigned int inout_derivative_size = 0;

  std::vector<std::shared_ptr<Var_Grad>> in_out;
  in_out.reserve(inout_dim.size());

  for (auto const &dim : inout_dim) {
    in_out.emplace_back(
      std::make_shared<Var_Grad>(dim, Tensor::Initializer::NONE, true, false,
                                 layer_name + std::to_string(cnt++)));
    if (is_act_layer || is_rnn_layer)
      inout_derivative_size += dim.getDataLen();
  }

  in_outs.push_back(in_out);
  is_act_type.push_back(is_act_layer);
  is_rnn_type.push_back(is_rnn_layer);
  is_flat_type.push_back(is_flat_layer);

  max_derivative_size = std::max(max_derivative_size, inout_derivative_size);
  return in_outs.back();
}

std::vector<std::shared_ptr<Var_Grad>> &
Manager::trackLayerOutputs(const std::string &layer_type,
                           const std::string &layer_name,
                           const std::vector<TensorDim> &output_dim,
                           const std::vector<TensorDim> &input_dim) {
  if (enable_inference_inout_memory_opt && input_dim.empty()) {
    throw std::invalid_argument(
      "Input dimensions are required with inference memory opt.");
  } else if (enable_inference_inout_memory_opt) {
    unsigned int shared_inout = 0;

    for (auto const &dim : output_dim)
      shared_inout += dim.getDataLen();

    for (auto const &dim : input_dim)
      shared_inout += dim.getDataLen();

    max_shared_inout = std::max(max_shared_inout, shared_inout);
  }

  return trackLayerInOuts(layer_type, layer_name + ":Output", output_dim);
}

std::vector<std::shared_ptr<Var_Grad>> &
Manager::trackLayerInputs(const std::string &layer_type,
                          const std::string &layer_name,
                          const std::vector<TensorDim> &input_dim,
                          const std::vector<TensorDim> &output_dim) {
  if (enable_inference_inout_memory_opt && output_dim.empty()) {
    throw std::invalid_argument(
      "Output dimensions are required with inference memory opt.");
  } else if (enable_inference_inout_memory_opt) {
    unsigned int shared_inout = 0;

    for (auto const &dim : output_dim)
      shared_inout += dim.getDataLen();

    for (auto const &dim : input_dim)
      shared_inout += dim.getDataLen();

    max_shared_inout = std::max(max_shared_inout, shared_inout);
  }

  return trackLayerInOuts(layer_type, layer_name + ":Input", input_dim);
}

void Manager::untrackVariable(const std::string &var_name) {
  for (unsigned int cnt = 0; cnt < in_outs.size(); cnt++) {
    if (!in_outs[cnt].empty() && in_outs[cnt][0]->getName() == var_name) {
      in_outs.erase(in_outs.begin() + cnt);
      is_act_type.erase(is_act_type.begin() + cnt);
      is_rnn_type.erase(is_rnn_type.begin() + cnt);
      is_flat_type.erase(is_flat_type.begin() + cnt);
      break;
    }
  }
}

void Manager::untrackLayerInOuts(const std::string &layer_name) {
  untrackVariable(layer_name + ":Input" + std::to_string(0));
  untrackVariable(layer_name + ":Output" + std::to_string(0));
}

void Manager::allocateInOuts() {
  /** Allocate the source tensors for shared memories */
  if (!shared_inout.empty())
    shared_inout.allocate();

  if (LAYER_V2) {
    for (auto &in : inputs_v2) {
      in->allocateVariable();
    }
    for (auto &out : outputs_v2) {
      out->allocateVariable();
    }
    for (auto &t : tensors_v2) {
      t->allocateVariable();
    }
  } else {
    for (auto &l_io : in_outs) {
      for (auto &io : l_io) {
        io->allocateVariable();
      }
    }
  }
}

void Manager::deallocateInOuts() {
  shared_inout.deallocate();

  if (LAYER_V2) {
    for (auto &in : inputs_v2) {
      in->deallocateVariable();
    }
    for (auto &out : outputs_v2) {
      out->deallocateVariable();
    }
    for (auto &t : tensors_v2) {
      t->deallocateVariable();
    }
  } else {
    for (auto &l_io : in_outs) {
      for (auto &io : l_io) {
        io->deallocateVariable();
      }
    }
  }
}

void Manager::allocateDerivatives() {
  /** Allocate the source tensors for shared memories */
  if (!shared_deriv.empty())
    shared_deriv.allocate();

  if (LAYER_V2) {
    for (auto &in : inputs_v2) {
      in->allocateGradient();
    }
    for (auto &out : outputs_v2) {
      out->allocateGradient();
    }
    for (auto &t : tensors_v2) {
      t->allocateGradient();
    }
  } else {
    for (auto &l_io : in_outs) {
      for (auto &io : l_io) {
        io->allocateGradient();
      }
    }
  }
}

void Manager::deallocateDerivatives() {
  shared_deriv.deallocate();

  if (LAYER_V2) {
    for (auto &in : inputs_v2) {
      in->deallocateGradient();
    }
    for (auto &out : outputs_v2) {
      out->deallocateGradient();
    }
    for (auto &t : tensors_v2) {
      t->deallocateGradient();
    }
  } else {
    for (auto &l_io : in_outs) {
      for (auto &io : l_io) {
        io->deallocateGradient();
      }
    }
  }
}

void Manager::initializeTensorsInference() {
  // @todo Do not count memory of the input tensor of the input layer and
  // output tensor of the last layer in the estimate of max_shared_inout as it
  // is not used

  // Initialize shared input/output memory for inference
  // @note Memory for label is not allocated here as inference doesnt has label
  if (enable_inference_inout_memory_opt)
    shared_inout = Tensor(TensorDim({max_shared_inout}), false);

  /**
   * A single buffer (shared_inout) provides memory for inputs and outputs of a
   * layer. Further, the output of layer i shares memory with input with layer
   * i+1. So, each alternate layer allocates memory from either the start of the
   * buffer or the end of the buffer, and use_first_last tracks this
   *
   * @note Label for the last layer is not initialized in inference.
   * @note Input for the first layer is not initialized in inference.
   */
  if (!LAYER_V2) {
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
  } else {
    // Inference Mode without optimizations
    for (auto &outs : outputs_v2) {
      outs->initialize(Tensor(), Tensor(), false);
    }

    // Inference Mode without optimizations
    for (auto &ts : tensors_v2) {
      ts->initialize(Tensor(), Tensor(), false);
    }

    // In inference mode, do not allocate the memory for the input of the first
    // layer. These is the first entry in the in_outs. Inference() will override
    // input tensors of the first layer
    for ([[maybe_unused]] auto &ins : inputs_v2) {
      // as inputs_v2 are only set for input layers, this can be skipped all the
      // way
      continue;
    }
  }
}

void Manager::initializeTensorsTrain() {
  // Initialize gradients
  initializeGradients();

  // Initialize shared derivative memory
  if (max_derivative_size > 0 && enable_activation_memory_opt)
    shared_deriv = Tensor(TensorDim({max_derivative_size}), false);

  if (!LAYER_V2) {
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
  } else {
    // Training Mode without optimizations
    for (auto &outs : outputs_v2) {
      outs->initialize(Tensor(), Tensor(), true);
    }

    // Training Mode without optimizations
    for (auto &ts : tensors_v2) {
      ts->initialize(Tensor(), Tensor(), true);
    }

    // Training Mode without optimizations
    for (auto &ins : inputs_v2) {
      ins->initialize(Tensor(), Tensor(), true);
    }
  }
}

/**
 * @brief Initialize the inputs/outputs/gradients/derivatives for the layer
 */
void Manager::initializeTensors(bool training) {
  // If weights not initialized, initialize weights as well
  if (!weights_initialized)
    initializeWeights();

  if (tensors_initialized && model_training == training)
    return;

  if (tensors_initialized)
    deinitializeTensors();

  model_training = training;
  if (model_training)
    initializeTensorsTrain();
  else
    initializeTensorsInference();
  tensors_initialized = true;
}

/**
 * @brief Deinitialize the inputs/outputs/gradients/derivatives for the layers
 */
void Manager::deinitializeTensors() {

  shared_deriv = Tensor();
  shared_inout = Tensor();
  shared_grad = Tensor();

  tensors_initialized = false;
}

/**
 * @brief     Create weights with the given spec
 *
 */
std::vector<Weight *>
Manager::requestWeights(const GraphNode &node,
                        const std::vector<Weight::Spec> &weights_spec) {
  auto ret = requestTensors<Weight>(node, weights_spec, weights_v2);
  const auto &exec_order = node.getExecutionOrder();
  for (auto const &w : ret) {
    auto const &vname = w->getName();
    auto const &gname = w->getGradientName();

    /** usage for weights */
    tensor_exec_order[vname].push_back(std::get<0>(exec_order));
    tensor_exec_order[vname].push_back(std::get<1>(exec_order));
    tensor_exec_order[vname].push_back(std::get<2>(exec_order));

    /** usage for its gradient only in calcGradient */
    tensor_exec_order[gname].push_back(std::get<1>(exec_order));

    /** set tensor lifespan */
    expandLifespan(vname, TensorLifespan::MAX_LIFESPAN);
    expandLifespan(gname, TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  }

  return ret;
}

/**
 * @brief     Create weights with the given spec
 *
 */
std::vector<Var_Grad *>
Manager::requestTensors(const GraphNode &node,
                        const std::vector<Var_Grad::Spec> &tensors_spec) {
  auto ret = requestTensors<Var_Grad>(node, tensors_spec, tensors_v2);
  const auto &exec_order = node.getExecutionOrder();
  for (unsigned int idx = 0; idx < ret.size(); idx++) {
    auto const &t = ret[idx];
    auto const &vname = t->getName();
    auto const &gname = t->getGradientName();
    auto const &tspan = std::get<4>(tensors_spec[idx]);

    /** usage for tensors */
    if (enum_class_logical_and<TensorLifespan>(
          tspan, TensorLifespan::FORWARD_FUNC_LIFESPAN))
      tensor_exec_order[vname].push_back(std::get<0>(exec_order));

    /** usage for tensors gradient in backwarding */
    if (enum_class_logical_and<TensorLifespan>(
          tspan, TensorLifespan::BACKWARD_FUNC_LIFESPAN)) {
      tensor_exec_order[vname].push_back(std::get<1>(exec_order));
      tensor_exec_order[gname].push_back(std::get<1>(exec_order));

      tensor_exec_order[vname].push_back(std::get<2>(exec_order));
      tensor_exec_order[gname].push_back(std::get<2>(exec_order));
    }

    /** set tensor lifespan */
    expandLifespan(vname, tspan);
    expandLifespan(gname, tspan);
  }

  return ret;
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestInputs(const GraphNode &node,
                       const std::vector<TensorDim> &inputs_dim,
                       const std::vector<std::string> &outputs_name) {

  auto const &tspan = TensorLifespan::ITERATION_LIFESPAN;
  std::vector<Var_Grad *> ret;

  if (outputs_name.empty()) {
    unsigned int count = 0;
    std::vector<Var_Grad::Spec> inputs_spec;

    std::transform(
      inputs_dim.begin(), inputs_dim.end(), std::back_inserter(inputs_spec),
      [&count, &node, &tspan](auto const &elem) {
        return std::make_tuple(elem, Tensor::Initializer::NONE, true,
                               node.getName() + std::string(":input") +
                                 std::to_string(count++),
                               tspan);
      });

    ret = requestTensors<Var_Grad>(node, inputs_spec, inputs_v2);
  } else {
    ret.reserve(inputs_dim.size());

    /**
     * Find already allocated output which must match the name and dimensions
     */
    for (unsigned int idx = 0; idx < inputs_dim.size(); idx++) {
      auto output_loc = name_map.at(outputs_name.at(idx));
      Var_Grad *vg = outputs_v2.at(output_loc).get();
      if (vg->getDim() != inputs_dim[idx])
        throw std::invalid_argument(
          "Dimension mismatch for the requested input");
      ret.push_back(vg);
    }
  }

  const auto &exec_order = node.getExecutionOrder();
  for (auto const &in : ret) {
    auto const &vname = in->getName();
    auto const &gname = in->getGradientName();

    /** usage for inputs */
    tensor_exec_order[vname].push_back(std::get<0>(exec_order));
    tensor_exec_order[vname].push_back(std::get<1>(exec_order));

    /** usage for inputs gradients (outgoing derivatives) */
    tensor_exec_order[gname].push_back(std::get<2>(exec_order));

    /** set tensor lifespan */
    expandLifespan(vname, tspan);
    expandLifespan(gname, tspan);
  }

  return ret;
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestOutputs(const GraphNode &node,
                        const std::vector<TensorDim> &outputs_dim) {
  unsigned int count = 0;
  auto const &tspan = TensorLifespan::ITERATION_LIFESPAN;
  std::vector<Var_Grad::Spec> outputs_spec;

  std::transform(
    outputs_dim.begin(), outputs_dim.end(), std::back_inserter(outputs_spec),
    [&count, &node, &tspan](auto const &elem) {
      return std::make_tuple(elem, Tensor::Initializer::NONE, true,
                             node.getName() + std::string(":output") +
                               std::to_string(count++),
                             tspan);
    });

  auto ret = requestTensors<Var_Grad>(node, outputs_spec, outputs_v2);
  const auto &exec_order = node.getExecutionOrder();
  for (auto const &out : ret) {
    auto const &vname = out->getName();
    auto const &gname = out->getGradientName();

    /** usage for outputs */
    tensor_exec_order[vname].push_back(std::get<0>(exec_order));

    /** usage for outputs gradients (incoming derivatives) */
    tensor_exec_order[gname].push_back(std::get<1>(exec_order));
    tensor_exec_order[gname].push_back(std::get<2>(exec_order));

    /**
     * TODO: below is needed only for activation layer as of now -
     * check if this can be worked around
     */
    tensor_exec_order[vname].push_back(std::get<2>(exec_order));

    /** set tensor lifespan */
    expandLifespan(vname, tspan);
    expandLifespan(gname, tspan);
  }

  return ret;
}

void Manager::expandLifespan(const std::string &name, TensorLifespan lifespan) {
  tensor_lifespan_map[name] =
    enum_class_or<TensorLifespan>(tensor_lifespan_map[name], lifespan);
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *> Manager::requestAllocatedOutputsAsInputs(
  const GraphNode &node, const std::vector<TensorDim> &inputs_dim,
  const std::vector<std::string> &outputs_name) {

  auto const &tspan = TensorLifespan::ITERATION_LIFESPAN;
  std::vector<Var_Grad *> ret;

  /** add the execution order and lifespan for the returning tensors */
  const auto &exec_order = node.getExecutionOrder();
  for (auto const &in : ret) {
    auto const &vname = in->getName();
    auto const &gname = in->getGradientName();

    /** usage for inputs */
    tensor_exec_order[vname].push_back(std::get<0>(exec_order));
    tensor_exec_order[vname].push_back(std::get<1>(exec_order));

    /** usage for inputs gradients (outgoing derivatives) */
    tensor_exec_order[gname].push_back(std::get<2>(exec_order));

    /** set tensor lifespan */
    expandLifespan(vname, tspan);
    expandLifespan(gname, tspan);
  }

  return ret;
}

std::vector<Weight *> Manager::getWeights() {
  std::vector<Weight *> all_weights;

  if (LAYER_V2) {
    for (auto &w : weights_v2) {
      all_weights.push_back(w.get());
    }
  } else {
    throw std::runtime_error("Using deprecated code. Upgrade to LayerV2");
  }

  return all_weights;
}

} // namespace nntrainer
