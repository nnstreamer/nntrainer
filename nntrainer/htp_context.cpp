// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 j2z0.lee <j2z0.lee@ax.samsung.com>
 *
 * @file   htp_context.cpp
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author j2z0.lee <j2z0.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP compute Context implementation.
 */

#ifdef ENABLE_HEXKL

#include <htp_context.h>

#include <app_context.h>
#include <compute_ops.h>
#include <context_data.h>
#include <cstdlib>
#include <cstring>
#include <htp_backend.h>
#include <mem_allocator.h>
#include <stdexcept>

namespace nntrainer {

namespace {
// Activation pools only: weight placement continues to use the CPU allocator.
class HtpActivationAllocator final : public MemAllocator {
public:
  void alloc(void **ptr, size_t size, size_t alignment) override {
    *ptr = HtpBackend::global().alloc_shared(size);
    if (!*ptr)
      throw std::bad_alloc();
    if (reinterpret_cast<uintptr_t>(*ptr) % alignment) {
      HtpBackend::global().free_shared(*ptr);
      *ptr = nullptr;
      throw std::runtime_error("HTP activation alignment is unsupported");
    }
    std::memset(*ptr, 0, size);
  }
  void free(void *ptr) override { HtpBackend::global().free_shared(ptr); }
  std::string getName() override { return "htp-rpcmem"; }
};
} // namespace

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const std::string &type,
                              const std::vector<std::string> &properties) {
  // HTP has no custom layers; reuse the CPU app context's factories.
  return AppContext::Global().createLayerObject(type, properties);
}

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const int int_key,
                              const std::vector<std::string> &properties) {
  return AppContext::Global().createLayerObject(int_key, properties);
}

void HtpContext::initialize() noexcept {
  // Ensure the CPU backend ops table exists — it is both the fallback
  // and the base class of HtpComputeOps. Go through ensureComputeOps()
  // rather than init_backend() directly: init_backend() is not
  // idempotent (__ggml_init, __openblas_set_num_threads), and
  // ensureComputeOps() is the call_once funnel every other Context
  // routes through.
  ensureComputeOps();

  auto cd = getContextData();
  if (!cd)
    return;

  // Wrap the NPU ops installation in try/catch so that a late exception from
  // HtpBackend::global() or get_htp_ops() (e.g., missing skel, driver error)
  // degrades to CPU rather than propagating out of this noexcept function
  // and hitting std::terminate.  Pattern mirrors ClContext::initialize().
  try {
    // NPU up -> HTP ops (shgemm on NPU, rest delegated to CPU via
    // inheritance). NPU down -> plain CPU ops, so an engine="htp" model
    // still runs, transparently, on the CPU.
    if (HtpBackend::global().enabled()) {
      cd->setComputeOps(get_htp_ops());
      const char *shared = std::getenv("NNTR_HTP_SHARED_ACTIVATIONS");
      if (shared && std::atoi(shared))
        cd->setMemAllocator(std::make_shared<HtpActivationAllocator>());

      // Unless opted in for activations above, model tensors use CPU memory.
      // The kernels stage into NPU-accessible (ION/rpcmem) buffers per call;
      // routing every model tensor through an NPU allocator exhausts the DMA
      // pool on a full transformer model.
    } else
      cd->setComputeOps(get_cpu_ops());
  } catch (...) {
    // HtpBackend::global() or get_htp_ops() threw — degrade to CPU.
    cd->setComputeOps(get_cpu_ops());
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
