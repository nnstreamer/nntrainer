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

#include <manager.h>
#include <nntrainer_log.h>

namespace nntrainer {
MMapedMemory::MMapedMemory(size_t size, bool allocate_fd) :
  fd(-1),
  buf(nullptr),
  buf_size(0) {

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
    if (fd_ != -1) {
      // unlink / close the given fd here
      close(fd_);
    }

    throw std::runtime_error("[MMapedMemory] mmap failed");
  }

  fd = fd_;
  buf = buf_;
  buf_size = size;

  ml_logd("[MMapedMemory] memory acquired size: %zu, fd: %d, addr: %p",
          buf_size, fd, buf);
}

MMapedMemory::~MMapedMemory() {
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

Manager::Manager(bool enable_gradient_memory_opt_, bool use_shared_memory_) :
  total_weight_size(0),
  total_grad_size(0),
  max_grad_size(0),
  enable_gradient_memory_opt(enable_gradient_memory_opt_),
  use_shared_memory(use_shared_memory_) {}

Manager::~Manager() {}

/**
 * @brief     Add weight to be tracked and updated with nntrainer
 */
void Manager::trackWeight(std::reference_wrapper<Weight> w) {
  /// @warning this does not track the weight size etcs.. This might break when
  /// use_shared_memory = true
  std::vector<std::reference_wrapper<Weight>> temp = {w};
  weights.emplace_back(temp);
}

/**
 * @brief     Add weights to be tracked and updated with nntrainer
 */
void Manager::trackWeights(std::vector<Weight> &ws) {
  std::vector<std::reference_wrapper<Weight>> layer_weights;
  layer_weights.reserve(ws.size());

  size_t weight_size = 0;
  size_t grad_size = 0;

  for (auto &w : ws) {
    layer_weights.emplace_back(std::ref(w));
    size_t len = w.getDim().getDataLen();
    weight_size += len;
    if (w.getTrainable())
      grad_size += len;
  }

  weights.push_back(layer_weights);

  total_weight_size += weight_size;
  total_grad_size += grad_size;
  max_grad_size = std::max(max_grad_size, grad_size);
}

/**
 * @brief Allocate and initialize the weight variable
 */
void Manager::initialize() {
  if (total_weight_size == 0) {
    ml_logw("Nothing done on initialize because there is no weight registered");
    return;
  }
  using AllocFunc = std::function<Tensor(const TensorDim &, size_t)>;

  AllocFunc allocate_none = [](const TensorDim &dim, size_t) {
    return Tensor();
  };

  AllocFunc allocate_weight = allocate_none;
  AllocFunc allocate_grad = allocate_none;

  if (use_shared_memory) {

    /// this creates memory and sets to @a memory and returns AllocFunc
    auto get_allocfunc = [](const size_t weight_size,
                            std::unique_ptr<MMapedMemory> &memory) {
      if (weight_size >= std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::invalid_argument(
          "weights exceed maximum size supported for shared memory");
      }
      size_t byte_size = weight_size * sizeof(float);
      memory = std::make_unique<MMapedMemory>(byte_size, true);
      return [&memory](const TensorDim &dim, size_t offset) {
        return Tensor::Map(memory->typedBuffer<float>(), dim, offset);
      };
    };

    allocate_weight = get_allocfunc(total_weight_size, weight_mmaped_memory);

    size_t grad_size =
      enable_gradient_memory_opt ? max_grad_size : total_grad_size;
    allocate_grad = get_allocfunc(grad_size, grad_mmaped_memory);

  } else {
    if (max_grad_size > 0 && enable_gradient_memory_opt) {
      std::shared_ptr<float> window(new float[max_grad_size],
                                    std::default_delete<float[]>());

      allocate_grad = [window](const TensorDim &dim, size_t offset) {
        return Tensor::Map(window, dim, offset);
      };
    }
  }

  size_t weight_offset = 0;
  size_t grad_offset = 0;

  for (auto &l_w : weights) {
    if (enable_gradient_memory_opt) {
      grad_offset = 0;
    }
    for (auto &w : l_w) {
      Weight &weight = w.get();
      auto dim = weight.getDim();
      Tensor weight_prealloc = allocate_weight(dim, weight_offset);
      Tensor grad_prealloc =
        weight.getTrainable() ? allocate_grad(dim, grad_offset) : Tensor();

      weight_offset += dim.getDataLen();
      grad_offset += dim.getDataLen();
      weight.initialize(weight_prealloc, grad_prealloc);
    }
  }
}

/**
 * @brief Track the inputs/ouputs of the layer
 */
void Manager::TrackLayerInOuts(const std::string layer_name,
                               const std::vector<TensorDim> &input_dim) {
  int cnt = 0;
  auto base_name = layer_name + ":Input";

  std::vector<std::shared_ptr<Var_Grad>> in_out;
  in_out.reserve(input_dim.size());

  for (auto const &dim : input_dim) {
    in_out.emplace_back(std::make_shared<Var_Grad>(
      dim, false, base_name + std::to_string(cnt++)));
  }

  in_outs.push_back(in_out);
}

} // namespace nntrainer
