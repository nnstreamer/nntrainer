// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels_fp16.cpp
 * @date	29 May 2024
 * @brief	Common blas OpenCL fp16 kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_strings.h>
#include <blas_kernels.h>
#include <unordered_map>
#include <mutex>
#include <CL/cl.h>
#include <functional>

namespace nntrainer {

// Result type for structured error handling
enum class BlasResult {
  SUCCESS = 0,
  KERNEL_REGISTRATION_FAILED,
  MEMORY_TRANSFER_FAILED,
  KERNEL_EXECUTION_FAILED,
  DEVICE_ERROR
};

// Global kernel cache for performance optimization
class KernelCache {
private:
  static std::unordered_map<std::string, ClContext::SharedPtrClKernel> cache_;
  static std::mutex cache_mutex_;

public:
  static ClContext::SharedPtrClKernel getOrCreateKernel(
    const std::string& kernel_source, const std::string& kernel_name) {
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = cache_.find(kernel_name);
    if (it != cache_.end()) {
      return it->second; // Return cached kernel
    }
    
    // Create and cache new kernel
    auto kernel = blas_cc->registerClKernel(kernel_source, kernel_name);
    if (kernel) {
      cache_[kernel_name] = kernel;
    }
    return kernel;
  }
  
  static void clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
  }
};

std::unordered_map<std::string, ClContext::SharedPtrClKernel> KernelCache::cache_;
std::mutex KernelCache::cache_mutex_;

// Asynchronous memory operation manager
class AsyncMemoryManager {
private:
  static cl_event events_[3]; // For input A, input B, output buffers
  
public:
  static bool writeDataAsync(ClBuffer* buffer, const cl_command_queue& queue,
                           size_t size, const void* data, int event_index) {
    return buffer->WriteDataRegionAsync(queue, size, data, &events_[event_index]);
  }
  
  static bool readDataAsync(ClBuffer* buffer, const cl_command_queue& queue,
                          size_t size, void* data, int event_index) {
    return buffer->ReadDataRegionAsync(queue, size, data, &events_[event_index]);
  }
  
  static bool waitForEvents(int count) {
    return clWaitForEvents(count, events_) == CL_SUCCESS;
  }
  
  static bool waitForEvent(int event_index) {
    return clWaitForEvents(1, &events_[event_index]) == CL_SUCCESS;
  }
  
  static void clearEvents() {
    for (int i = 0; i < 3; i++) {
      if (events_[i]) clReleaseEvent(events_[i]);
    }
  }
};

cl_event AsyncMemoryManager::events_[3];

// Device capability manager for adaptive optimization
class DeviceCapabilityManager {
private:
  static bool initialized_;
  static size_t max_work_group_size_;
  static size_t max_compute_units_;
  static size_t max_local_memory_;
  static bool supports_fp16_;
  static int optimal_tile_size_;
  
public:
  static bool initialize(cl_device_id device) {
    if (initialized_) return true;
    
    // Query device capabilities
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                   sizeof(max_work_group_size_), &max_work_group_size_, nullptr);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                   sizeof(max_compute_units_), &max_compute_units_, nullptr);
    
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                   sizeof(max_local_memory_), &max_local_memory_, nullptr);
    
    // Check FP16 support
    size_t extensions_size;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensions_size);
    std::string extensions(extensions_size, '\0');
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensions_size, 
                   &extensions[0], nullptr);
    
    supports_fp16_ = extensions.find("cl_khr_fp16") != std::string::npos;
    
    // Determine optimal tile size based on device vendor
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::string name(device_name);
    
    if (name.find("NVIDIA") != std::string::npos) {
      optimal_tile_size_ = 32; // NVIDIA GPUs prefer 32x32 tiles
    } else if (name.find("AMD") != std::string::npos) {
      optimal_tile_size_ = 16; // AMD GPUs prefer 16x16 tiles
    } else if (name.find("Intel") != std::string::npos) {
      optimal_tile_size_ = 8;  // Intel integrated GPUs prefer smaller tiles
    } else {
      optimal_tile_size_ = 16; // Default
    }
    
    initialized_ = true;
    return true;
  }
  
  static size_t getMaxWorkGroupSize() { return max_work_group_size_; }
  static size_t getMaxComputeUnits() { return max_compute_units_; }
  static size_t getMaxLocalMemory() { return max_local_memory_; }
  static bool supportsFP16() { return supports_fp16_; }
  static int getOptimalTileSize() { return optimal_tile_size_; }
};

// Static member definitions
bool DeviceCapabilityManager::initialized_ = false;
size_t DeviceCapabilityManager::max_work_group_size_ = 256;
size_t DeviceCapabilityManager::max_compute_units_ = 16;
size_t DeviceCapabilityManager::max_local_memory_ = 32768;
bool DeviceCapabilityManager::supports_fp16_ = false;
int DeviceCapabilityManager::optimal_tile_size_ = 16;

// RAII-based operation manager for clean resource handling
class BlasOperationManager {
private:
  bool resources_acquired_;
  
public:
  BlasOperationManager() : resources_acquired_(false) {}
  
  ~BlasOperationManager() {
    // Automatic cleanup if needed
  }
  
  BlasResult executeOperation(std::function<BlasResult()> operation) {
    try {
      return operation();
    } catch (...) {
      return BlasResult::DEVICE_ERROR;
    }
  }
};

// Dynamic work group configuration
struct WorkGroupConfig {
  int global_size[3];
  int local_size[3];
};

// Calculate optimal work group sizes based on operation and dimensions
WorkGroupConfig calculateOptimalWorkGroup(unsigned int dim1, unsigned int dim2, 
                                         const std::string& operation, unsigned int dim3 = 1) {
  WorkGroupConfig config;
  
  // Query device capabilities (should be cached globally)
  size_t max_work_group_size = DeviceCapabilityManager::getMaxWorkGroupSize();
  size_t max_compute_units = DeviceCapabilityManager::getMaxComputeUnits();
  size_t max_local_memory = DeviceCapabilityManager::getMaxLocalMemory();
  int optimal_tile = DeviceCapabilityManager::getOptimalTileSize();
  
  if (operation == "sgemv") {
    // For GEMV: optimize for memory bandwidth
    size_t optimal_local = std::min(static_cast<size_t>(64), max_work_group_size);
    
    config.global_size[0] = ((dim1 + optimal_local - 1) / optimal_local) * optimal_local;
    config.global_size[1] = 1;
    config.global_size[2] = 1;
    
    config.local_size[0] = optimal_local;
    config.local_size[1] = 1;
    config.local_size[2] = 1;
  } else if (operation == "sgemm") {
    // For GEMM: optimize for compute throughput with device-adaptive tiling
    int TILE_SIZE = optimal_tile;
    
    // Ensure we don't exceed max work group size
    while (TILE_SIZE * TILE_SIZE > max_work_group_size) {
      TILE_SIZE /= 2;
    }
    
    // Minimum viable tile size
    TILE_SIZE = std::max(TILE_SIZE, 4);
    
    config.local_size[0] = TILE_SIZE;
    config.local_size[1] = TILE_SIZE;
    config.local_size[2] = 1;
    
    config.global_size[0] = ((dim2 + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    config.global_size[1] = ((dim1 + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    config.global_size[2] = 1;
  } else if (operation == "dot") {
    // For DOT: optimize for reduction operations
    size_t reduction_size = std::min(static_cast<size_t>(128), max_work_group_size);
    
    config.local_size[0] = reduction_size;
    config.local_size[1] = 1;
    config.local_size[2] = 1;
    
    config.global_size[0] = ((dim1 + reduction_size - 1) / reduction_size) * reduction_size;
    config.global_size[1] = 1;
    config.global_size[2] = 1;
  } else if (operation == "vector") {
    // For vector operations (addition, sscal): optimize for memory bandwidth
    size_t vector_size = std::min(static_cast<size_t>(64), max_work_group_size);
    
    config.local_size[0] = vector_size;
    config.local_size[1] = 1;
    config.local_size[2] = 1;
    
    config.global_size[0] = ((dim1 + vector_size - 1) / vector_size) * vector_size;
    config.global_size[1] = 1;
    config.global_size[2] = 1;
  } else {
    // Default fallback
    config.global_size[0] = dim1;
    config.global_size[1] = 1;
    config.global_size[2] = 1;
    config.local_size[0] = 1;
    config.local_size[1] = 1;
    config.local_size[2] = 1;
  }
  
  return config;
}

// Optimized sgemv with structured error handling
BlasResult sgemv_cl_optimized(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {

  BlasOperationManager manager;
  
  return manager.executeOperation([&]() -> BlasResult {
    // Get cached kernel
    ClContext::SharedPtrClKernel kernel_ptr;
    if (TransA) {
      kernel_ptr = KernelCache::getOrCreateKernel(
        getHgemvClKernel(), "sgemv_cl_fp16");
    } else {
      kernel_ptr = KernelCache::getOrCreateKernel(
        getHgemvClNoTransKernel(), "sgemv_cl_noTrans_fp16");
    }
    
    if (!kernel_ptr) {
      return BlasResult::KERNEL_REGISTRATION_FAILED;
    }

    // Memory operations
    size_t dim1_size = sizeof(_FP16) * dim1;
    size_t dim2_size = sizeof(_FP16) * dim2;
    size_t matrix_size = dim1 * dim2 * sizeof(_FP16);

    // Async memory transfers
    if (!AsyncMemoryManager::writeDataAsync(clbuffInstance.getInBufferA(), 
                                           blas_cc->command_queue_inst_, matrix_size, matAdata, 0) ||
        !AsyncMemoryManager::writeDataAsync(clbuffInstance.getInBufferB(), 
                                           blas_cc->command_queue_inst_, dim2_size, vecXdata, 1)) {
      return BlasResult::MEMORY_TRANSFER_FAILED;
    }

    // Wait for input transfers
    if (!AsyncMemoryManager::waitForEvents(2)) {
      return BlasResult::MEMORY_TRANSFER_FAILED;
    }

    // Set kernel arguments efficiently
    if (!kernel_ptr->SetKernelArguments(0, clbuffInstance.getInBufferA(), sizeof(cl_mem)) ||
        !kernel_ptr->SetKernelArguments(1, clbuffInstance.getInBufferB(), sizeof(cl_mem)) ||
        !kernel_ptr->SetKernelArguments(2, clbuffInstance.getOutBufferA(), sizeof(cl_mem)) ||
        !kernel_ptr->SetKernelArguments(3, &dim2, sizeof(int)) ||
        !kernel_ptr->SetKernelArguments(4, &lda, sizeof(int))) {
      return BlasResult::KERNEL_EXECUTION_FAILED;
    }

    // Execute with optimized work groups
    WorkGroupConfig wg_config = calculateOptimalWorkGroup(dim1, dim2, "sgemv");
    if (!blas_cc->command_queue_inst_.DispatchCommand(kernel_ptr, wg_config.global_size, wg_config.local_size)) {
      return BlasResult::KERNEL_EXECUTION_FAILED;
    }

    // Async result reading
    if (!AsyncMemoryManager::readDataAsync(clbuffInstance.getOutBufferA(), 
                                          blas_cc->command_queue_inst_, dim1_size, vecYdata, 0)) {
      return BlasResult::MEMORY_TRANSFER_FAILED;
    }

    // Wait for completion
    if (!AsyncMemoryManager::waitForEvent(0)) {
      return BlasResult::MEMORY_TRANSFER_FAILED;
    }

    return BlasResult::SUCCESS;
  });
}

// Wrapper function for backward compatibility
void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {
  BlasResult result = sgemv_cl_optimized(matAdata, vecXdata, vecYdata, TransA, dim1, dim2, lda);
  // Could add logging or error handling based on result
}

// Legacy implementation for reference (to be removed after validation)
void sgemv_cl_legacy(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemv_fp16_ptr;

    if (TransA) {
      kernel_sgemv_fp16_ptr = KernelCache::getOrCreateKernel(
        getHgemvClKernel(), "sgemv_cl_fp16");
    } else {
      kernel_sgemv_fp16_ptr = KernelCache::getOrCreateKernel(
        getHgemvClNoTransKernel(), "sgemv_cl_noTrans_fp16");
    }

    if (!kernel_sgemv_fp16_ptr) {
      break;
    }

    size_t dim1_size = sizeof(_FP16) * dim1;
    size_t dim2_size = sizeof(_FP16) * dim2;
    size_t matrix_size = dim1 * dim2 * sizeof(_FP16);

    // OPTIMIZED: Asynchronous memory transfers
    result = AsyncMemoryManager::writeDataAsync(
      clbuffInstance.getInBufferA(), blas_cc->command_queue_inst_, 
      matrix_size, matAdata, 0);
    if (!result) {
      break;
    }

    result = AsyncMemoryManager::writeDataAsync(
      clbuffInstance.getInBufferB(), blas_cc->command_queue_inst_, 
      dim2_size, vecXdata, 1);
    if (!result) {
      break;
    }

    // Wait for input transfers before kernel execution
    if (!AsyncMemoryManager::waitForEvents(2)) break;

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(
      1, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(
      2, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    // OPTIMIZED: Dynamic work group calculation
    WorkGroupConfig wg_config = calculateOptimalWorkGroup(dim1, dim2, "sgemv");
    const int work_groups_count[3] = {wg_config.global_size[0], wg_config.global_size[1], wg_config.global_size[2]};
    const int work_group_size[3] = {wg_config.local_size[0], wg_config.local_size[1], wg_config.local_size[2]};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_sgemv_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // OPTIMIZED: Asynchronous result reading
    result = AsyncMemoryManager::readDataAsync(
      clbuffInstance.getOutBufferA(), blas_cc->command_queue_inst_, 
      dim1_size, vecYdata, 0);
    if (!result) {
      break;
    }

    // Wait for result transfer to complete
    AsyncMemoryManager::waitForEvent(0);

  } while (false);
}

_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1) {

  bool result = false;

  _FP16 cl_ret = 0;

  do {
    ClContext::SharedPtrClKernel kernel_dot_fp16_ptr =
      KernelCache::getOrCreateKernel(getDotClKernelFP16(), "dot_cl_fp16");

    if (!kernel_dot_fp16_ptr) {
      break;
    }

    size_t dim1_size = sizeof(_FP16) * dim1;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecXdata);
    if (!result) {
      break;
    }

    result = kernel_dot_fp16_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16_ptr->SetKernelArguments(
      1, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16_ptr->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16_ptr->SetKernelArguments(
      3, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    // OPTIMIZED: Dynamic work group calculation for dot product
    WorkGroupConfig wg_config = calculateOptimalWorkGroup(dim1, 1, "dot");
    const int work_groups_count[3] = {wg_config.global_size[0], wg_config.global_size[1], wg_config.global_size[2]};
    const int work_group_size[3] = {wg_config.local_size[0], wg_config.local_size[1], wg_config.local_size[2]};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_dot_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, sizeof(_FP16), &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {

  std::string kernel_func_;
  std::string sgemm_cl_kernel_fp16_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClNoTransKernel();
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClTransAKernel();
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClTransBKernel();
  } else {
    kernel_func_ = "sgemm_cl_transAB_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClTransABKernel();
  }

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemm_fp16_ptr =
      KernelCache::getOrCreateKernel(sgemm_cl_kernel_fp16_, kernel_func_);
    if (!kernel_sgemm_fp16_ptr) {
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(_FP16);
    size_t k_n_size = K * N * sizeof(_FP16);
    size_t m_n_size = M * N * sizeof(_FP16);

    // OPTIMIZED: Asynchronous memory transfers
    result = AsyncMemoryManager::writeDataAsync(
      clbuffInstance.getInBufferA(), blas_cc->command_queue_inst_, 
      m_k_size, A, 0);
    if (!result) {
      break;
    }

    result = AsyncMemoryManager::writeDataAsync(
      clbuffInstance.getInBufferB(), blas_cc->command_queue_inst_, 
      k_n_size, B, 1);
    if (!result) {
      break;
    }

    // Wait for input data transfers
    if (!AsyncMemoryManager::waitForEvents(2)) break;

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, m_n_size, C);
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(
      1, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(
      2, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(3, &M, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(4, &N, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(5, &K, sizeof(int));
    if (!result) {
      break;
    }

    // OPTIMIZED: Dynamic work group calculation for matrix multiplication
    WorkGroupConfig wg_config = calculateOptimalWorkGroup(M, N, "sgemm", K);
    const int work_groups_count[3] = {wg_config.global_size[0], wg_config.global_size[1], wg_config.global_size[2]};
    const int work_group_size[3] = {wg_config.local_size[0], wg_config.local_size[1], wg_config.local_size[2]};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_sgemm_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, m_n_size, C);
    if (!result) {
      break;
    }

  } while (false);
}

void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_addition_fp16_ptr =
      KernelCache::getOrCreateKernel(getAdditionClKernelFP16(), "addition_cl_fp16");
    if (!kernel_addition_fp16_ptr) {
      break;
    }

    size_t dim1_size = sizeof(_FP16) * size_input;
    size_t dim2_size = sizeof(_FP16) * size_res;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, input);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);
    if (!result) {
      break;
    }

    result = kernel_addition_fp16_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_addition_fp16_ptr->SetKernelArguments(
      1, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_addition_fp16_ptr->SetKernelArguments(2, &size_input, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_addition_fp16_ptr->SetKernelArguments(3, &size_res, sizeof(int));
    if (!result) {
      break;
    }

    // OPTIMIZED: Dynamic work group calculation for vector addition
    WorkGroupConfig wg_config = calculateOptimalWorkGroup(size_res, 1, "vector");
    const int work_groups_count[3] = {wg_config.global_size[0], wg_config.global_size[1], wg_config.global_size[2]};
    const int work_group_size[3] = {wg_config.local_size[0], wg_config.local_size[1], wg_config.local_size[2]};
    
    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_addition_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);

    if (!result) {
      break;
    }

  } while (false);
}

void sscal_cl(_FP16 *X, const unsigned int N, const float alpha) {
  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sscal_fp16_ptr =
      KernelCache::getOrCreateKernel(getHscalClKernel(), "sscal_cl_fp16");

    if (!kernel_sscal_fp16_ptr) {
      break;
    }

    size_t x_size = N * sizeof(_FP16);

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, x_size, X);
    if (!result) {
      break;
    }

    result = kernel_sscal_fp16_ptr->SetKernelArguments(
      0, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_sscal_fp16_ptr->SetKernelArguments(1, &alpha, sizeof(float));
    if (!result) {
      break;
    }

    // OPTIMIZED: Dynamic work group calculation for vector scaling
    WorkGroupConfig wg_config = calculateOptimalWorkGroup(N, 1, "vector");
    const int work_groups_count[3] = {wg_config.global_size[0], wg_config.global_size[1], wg_config.global_size[2]};
    const int work_group_size[3] = {wg_config.local_size[0], wg_config.local_size[1], wg_config.local_size[2]};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_sscal_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, x_size, X);
    if (!result) {
      break;
    }

  } while (false);
}

void transpose_cl_axis(const _FP16 *in, _FP16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_transpose_fp_16_ptr;
    switch (axis) {
    case 0:
      kernel_transpose_fp_16_ptr = KernelCache::getOrCreateKernel(
        getTransposeClAxis0KernelFP16(), "transpose_cl_fp16_axis0");
      break;
    case 1:
      kernel_transpose_fp_16_ptr = KernelCache::getOrCreateKernel(
        getTransposeClAxis1KernelFP16(), "transpose_cl_fp16_axis1");
      break;
    case 2:
      kernel_transpose_fp_16_ptr = KernelCache::getOrCreateKernel(
        getTransposeClAxis2KernelFP16(), "transpose_cl_fp16_axis2");
      break;
    default:
      throw std::invalid_argument("failed to register CL kernel");
      break;
    }
    if (!kernel_transpose_fp_16_ptr) {
      break;
    }

    size_t dim_size = sizeof(_FP16) * input_batch_size * input_height *
                      input_width * input_channels;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim_size, in);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim_size, res);
    if (!result) {
      break;
    }

    result = kernel_transpose_fp_16_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_fp_16_ptr->SetKernelArguments(
      1, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_fp_16_ptr->SetKernelArguments(
      2, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_transpose_fp_16_ptr->SetKernelArguments(3, &input_channels,
                                                            sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_transpose_fp_16_ptr->SetKernelArguments(4, &input_height,
                                                            sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_transpose_fp_16_ptr->SetKernelArguments(5, &input_width,
                                                            sizeof(int));
    if (!result) {
      break;
    }

    int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
    if (axis == 2)
      work_groups_count[0] = (int)input_channels;

    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_transpose_fp_16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim_size, res);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
