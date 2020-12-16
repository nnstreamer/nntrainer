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

Manager::Manager(bool enable_gradient_memory_opt_, bool use_shared_memory_) :
  total_weight_size(0),
  total_grad_size(0),
  max_grad_size(0),
  enable_gradient_memory_opt(enable_gradient_memory_opt_),
  use_shared_memory(use_shared_memory_),
  fd(-1),
  buf(nullptr),
  buf_size(0) {}

Manager::~Manager() { releaseSharedMemory(); }

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
    size_t grad_size =
      enable_gradient_memory_opt ? max_grad_size : total_grad_size;
    size_t total_size = total_weight_size + grad_size;

    if (total_size >= std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::invalid_argument(
        "weights exceed maximum size supported for shared memory");
    }

    size_t weight_bytes_size =
      (total_weight_size + total_grad_size) * sizeof(float);
    initializeSharedMemory(weight_bytes_size);

    allocate_grad = allocate_weight = [&](const TensorDim &dim, size_t offset) {
      return Tensor::Wrap(buf, dim, offset);
    };

  } else {
    if (max_grad_size > 0 && enable_gradient_memory_opt) {
      std::shared_ptr<float> window(new float[max_grad_size],
                                    std::default_delete<float[]>());

      allocate_grad = [window](const TensorDim &dim, size_t offset) {
        return Tensor::Wrap(window, dim, offset);
      };
    }
  }

  size_t weight_offset = 0;
  size_t grad_initial_offset = use_shared_memory ? total_weight_size : 0;
  size_t grad_offset = grad_initial_offset;

  for (auto &l_w : weights) {
    if (enable_gradient_memory_opt) {
      grad_offset = grad_initial_offset;
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

void Manager::initializeSharedMemory(size_t size) {
  if (buf != nullptr || fd > 0 || buf_size > 0) {
    throw std::runtime_error("[Manager] manager is already holding a buffer");
  }

#ifdef __ANDROID__
  /// unfortunately, memfd_create is not supported before android level 30
  auto fd_ = ASharedMemory_create("", size);
  if (fd_ < 0) {
    releaseSharedMemory();
    throw std::runtime_error("[Manager] creating mem fd failed");
  }

  if (ASharedMemory_setProt(fd_, PROT_READ | PROT_WRITE) < 0) {
    releaseSharedMemory();
    throw std::runtime_error("[Manager] Setting prot failed");
  }

  auto buf_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
#else
  /// @todo create a file in tmpfs and bind to memfs
  /// memfd_create is not available for number of platforms so this is commented
  // auto fd_ = memfd_create("", 0);
  // if (fd_ < 0) {
  //   releaseSharedMemory();
  //   throw std::runtime_error("[Manager] creating mem fd failed");
  // }
  // if (ftruncate(fd_, size) < 0) {
  //   releaseSharedMemory();
  //   throw std::runtime_error("[Manager] truncating fd failed");
  // }

  auto fd_ = -1;
  auto buf_ = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, fd_, 0);
#endif
  if (buf_ == MAP_FAILED) {
    releaseSharedMemory();
    throw std::runtime_error("[Manager] mmap failed");
  }

  buf = reinterpret_cast<float *>(buf_);
  fd = fd_;
  buf_size = size;

  ml_logd("[Manager] memory acquired size: %zu, fd: %d, addr: %p", buf_size, fd,
          buf);
}

void Manager::releaseSharedMemory() noexcept {
  if (buf == nullptr) {
    ml_logd("[Manager] buf is already empty, not released");
    return;
  }

#ifdef DEBUG
  assert(buf_size > 0 && fd > 0);
#endif
  if (munmap(buf, buf_size) < 0) {
    ml_logw("[Manager] munmap failed on destruction please check");
  }

  if (fd != -1) {
    if (close(fd) < 0) {
      ml_logw("[Manager] closing fd failed on destruction please check");
    }
  }

  fd = -1;
  buf = nullptr;
  buf_size = 0;
  ml_logd("[Manager] buf released");
}

} // namespace nntrainer
