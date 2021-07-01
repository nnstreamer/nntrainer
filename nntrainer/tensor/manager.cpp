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
#include <flatten_layer.h>
#include <layer_node.h>
#include <manager.h>
#include <nntrainer_log.h>
#include <rnn.h>

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

/**
 * @brief Allocate and initialize the weight variable
 */
void Manager::initializeWeights() {
  if (total_weight_size == 0) {
    ml_logw("Nothing done on initialize because there is no weight registered");
    return;
  }

  if (weights_initialized)
    return;

  AllocFunc allocate_weight = getAllocFunc(true);

  unsigned int weight_offset = 0;

  if (LAYER_V2) {
    for (auto &l_w : weights_v2) {
      for (auto &w : l_w) {
        const auto &dim = w->getDim();

        Tensor weight_prealloc = allocate_weight(dim, weight_offset);
        weight_offset += dim.getDataLen();

        w->initializeVariable(weight_prealloc);
      }
    }
  } else {
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
    for (auto &l_w : weights_v2) {
      for (auto &w : l_w) {
        w->allocateVariable();
      }
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
    for (auto &l_w : weights_v2) {
      for (auto &w : l_w) {
        w->deallocateVariable();
      }
    }
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
  if (!shared_grad.uninitialized())
    shared_grad.allocate();

  if (LAYER_V2) {
    for (auto &l_w : weights_v2) {
      for (auto &w : l_w) {
        w->allocateGradient();
      }
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
    for (auto &l_w : weights_v2) {
      for (auto &w : l_w) {
        w->deallocateGradient();
      }
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
  if (total_weight_size == 0) {
    ml_logw("Nothing done on initialize because there is no weight registered");
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
    in_out.emplace_back(std::make_shared<Var_Grad>(
      dim, true, false, layer_name + std::to_string(cnt++)));
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
  if (!shared_inout.uninitialized())
    shared_inout.allocate();

  if (LAYER_V2) {
    for (auto &li : inputs_v2) {
      for (auto &in : li) {
        in->allocateVariable();
      }
    }
    for (auto &lo : outputs_v2) {
      for (auto &out : lo) {
        out->allocateVariable();
      }
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
    for (auto &li : inputs_v2) {
      for (auto &in : li) {
        in->deallocateVariable();
      }
    }
    for (auto &lo : outputs_v2) {
      for (auto &out : lo) {
        out->deallocateVariable();
      }
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
  if (!shared_deriv.uninitialized())
    shared_deriv.allocate();

  if (LAYER_V2) {
    for (auto &li : inputs_v2) {
      for (auto &in : li) {
        in->allocateGradient();
      }
    }
    for (auto &lo : outputs_v2) {
      for (auto &out : lo) {
        out->allocateGradient();
      }
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
    for (auto &li : inputs_v2) {
      for (auto &in : li) {
        in->deallocateGradient();
      }
    }
    for (auto &lo : outputs_v2) {
      for (auto &out : lo) {
        out->deallocateGradient();
      }
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
    for (auto &layer_outs : outputs_v2) {
      // TODO:For flatten layer, do not assign new memory

      for (auto &outs : layer_outs) {
        outs->initialize(Tensor(), Tensor(), false);
      }
    }

    // Inference Mode without optimizations
    for (auto &layer_ts : tensors_v2) {
      for (auto &ts : layer_ts) {
        ts->initialize(Tensor(), Tensor(), false);
      }
    }

    // In inference mode, do not allocate the memory for the input of the first
    // layer. These is the first entry in the in_outs. Inference() will override
    // input tensors of the first layer
    for ([[maybe_unused]] auto &layer_ins : inputs_v2) {
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
    for (auto &layer_outs : outputs_v2) {
      for (auto &outs : layer_outs) {
        outs->initialize(Tensor(), Tensor(), true);
      }
    }

    // Training Mode without optimizations
    for (auto &layer_ts : tensors_v2) {
      for (auto &ts : layer_ts) {
        ts->initialize(Tensor(), Tensor(), true);
      }
    }

    // Training Mode without optimizations
    for (auto &layer_ins : inputs_v2) {
      for (auto &ins : layer_ins) {
        ins->initialize(Tensor(), Tensor(), true);
      }
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
  return requestTensors<Weight>(node, weights_spec, weights_v2);
}

/**
 * @brief     Create weights with the given spec
 *
 */
std::vector<Var_Grad *>
Manager::requestTensors(const GraphNode &node,
                        const std::vector<Var_Grad::Spec> &tensors_spec) {
  return requestTensors<Var_Grad>(node, tensors_spec, tensors_v2);
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestInputs(const GraphNode &node,
                       const std::vector<TensorDim> &inputs_dim) {
  unsigned int count = 0;
  std::vector<Var_Grad::Spec> inputs_spec;
  std::transform(
    inputs_dim.begin(), inputs_dim.end(), std::back_inserter(inputs_spec),
    [&count, &node](auto const &elem) {
      return std::make_tuple(elem, true,
                             node.getName() + std::string("_input") +
                               std::to_string(count++));
    });
  return requestTensors<Var_Grad>(node, inputs_spec, inputs_v2);
}

/**
 * @brief     Create tensors with the given spec
 */
std::vector<Var_Grad *>
Manager::requestOutputs(const GraphNode &node,
                        const std::vector<TensorDim> &outputs_dim) {
  unsigned int count = 0;
  std::vector<Var_Grad::Spec> outputs_spec;
  std::transform(
    outputs_dim.begin(), outputs_dim.end(), std::back_inserter(outputs_spec),
    [&count, &node](auto const &elem) {
      return std::make_tuple(elem, true,
                             node.getName() + std::string("_output") +
                               std::to_string(count++));
    });
  return requestTensors<Var_Grad>(node, outputs_spec, outputs_v2);
}

void Manager::requestOptimizerVariable(
  std::function<std::vector<TensorDim>(const TensorDim &)> cb,
  bool request_only_trainable) {
  if (LAYER_V2) {
    for (auto &weight_v2 : weights_v2) {
      for (auto &w : weight_v2) {
        if (request_only_trainable && !w->needsGradient()) {
          continue;
        }

        std::vector<TensorDim> dims = cb(w->getDim());
        for (auto &dim : dims) {
          w->addOptimizerVariable(dim);
        }
      }
    }
  } else {
    for (auto &weight : weights) {
      for (auto &rw_w : weight) {
        auto &w = rw_w.get();
        if (request_only_trainable && !w.needsGradient()) {
          continue;
        }

        const TensorDim &dim = w.getDim();
        std::vector<TensorDim> dims = cb(dim);
        for (auto &dim : dims) {
          w.addOptimizerVariable(dim);
        }
      }
    }
  }
}

} // namespace nntrainer
